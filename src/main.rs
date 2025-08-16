#![recursion_limit = "256"]
/// VAE in burn
pub mod data;
use crate::data::MnistBatcher;
use burn::backend::Wgpu;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::MnistDataset;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::NoStdTrainingRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{TrainOutput, TrainStep, ValidStep};
use data::MnistBatch;
use image::{ImageBuffer, Luma};

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
  layers: Vec<nn::Linear<B>>,
  activation: nn::Relu,
}

impl<B: Backend> MLP<B> {
  pub fn new(layers: Vec<nn::Linear<B>>, activation: nn::Relu) -> Self {
    Self { layers, activation }
  }

  pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let mut x = input;
    let num_layers = self.layers.len();
    for idx in 0..(num_layers - 1) {
      x = self.layers[idx].forward(x);
      x = self.activation.forward(x);
    }
    x = self.layers[num_layers - 1].forward(x);
    x
  }
}

#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
  enc: MLP<B>,
  dec: MLP<B>,
  enc_mu: nn::Linear<B>,
  enc_logvar: nn::Linear<B>,
  kld_weight: f32,
}

const SIZES: [usize; 4] = [784, 128, 64, 32];

impl<B: Backend> Default for VAE<B> {
  fn default() -> Self {
    let device = B::Device::default();
    Self::new(&device, 1e-1)
  }
}

impl<B: Backend> VAE<B> {
  pub fn new(device: &B::Device, kld_weight: f32) -> Self {
    let encoder_layers = SIZES
      .windows(2)
      .map(|window| nn::LinearConfig::new(window[0], window[1]).init(device))
      .collect::<Vec<_>>();
    let decoder_layers = SIZES
      .iter()
      .rev()
      .collect::<Vec<&usize>>()
      .windows(2)
      .map(|window| nn::LinearConfig::new(*window[0], *window[1]).init(device))
      .collect::<Vec<_>>();

    let enc_mu = nn::LinearConfig::new(SIZES.last().copied().unwrap_or(32), 32).init(device);
    let enc_logvar = nn::LinearConfig::new(SIZES.last().copied().unwrap_or(32), 32).init(device);

    Self {
      enc: MLP::new(encoder_layers, nn::Relu),
      dec: MLP::new(decoder_layers, nn::Relu),
      enc_mu,
      enc_logvar,
      kld_weight,
    }
  }

  pub fn encode(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let hidden = self.enc.forward(input);
    let hidden = nn::Relu.forward(hidden); // apply non-linearity
    let mu = self.enc_mu.forward(hidden.clone());
    let logvar = self.enc_logvar.forward(hidden);
    (mu, logvar)
  }

  pub fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
    let std = logvar.exp().sqrt();
    let eps = mu.random_like(burn::tensor::Distribution::Normal(0.0, 1.0));
    mu + std * eps
  }

  pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
    self.dec.forward(z)
  }

  pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let (mu, logvar) = self.encode(input);
    let z = self.reparameterize(mu.clone(), logvar.clone());
    (mu, logvar, self.decode(z))
  }

  pub fn forward_vae(&self, input: Tensor<B, 2>) -> VaeOutput<B> {
    let (mu, logvar, recon) = self.forward(input.clone());
    let recon_loss = burn::nn::loss::MseLoss::new().forward(
      recon.clone(),
      input.clone(),
      nn::loss::Reduction::Mean,
    );
    let kld_loss = (-0.5f32
      * (1.0f32 + logvar.clone() - mu.clone().powf_scalar(2.0f32) - logvar.clone().exp()))
    .mean();
    let kld_loss = kld_loss * 1e-1;
    VaeOutput {
      loss: recon_loss.clone() + kld_loss.clone(),
      recon_loss,
      kld_loss,
      recon,
      input,
    }
  }
}

#[derive(Debug)]
pub struct VaeOutput<B: Backend> {
  pub loss: Tensor<B, 1>,       // total loss
  pub recon_loss: Tensor<B, 1>, // reconstruction loss
  pub kld_loss: Tensor<B, 1>,   // KL divergence loss
  pub recon: Tensor<B, 2>,      // reconstructed images
  pub input: Tensor<B, 2>,      // original images
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, VaeOutput<B>> for VAE<B> {
  fn step(&self, item: MnistBatch<B>) -> TrainOutput<VaeOutput<B>> {
    let output = self.forward_vae(item.images);
    TrainOutput::new(self, output.loss.backward(), output)
  }
}

impl<B: AutodiffBackend> ValidStep<MnistBatch<B>, VaeOutput<B>> for VAE<B> {
  fn step(&self, item: MnistBatch<B>) -> VaeOutput<B> {
    self.forward_vae(item.images)
  }
}

#[derive(Config)]
pub struct MnistTrainingConfig {
  #[config(default = 30)]
  pub num_epochs: usize,

  #[config(default = 64)]
  pub batch_size: usize,

  #[config(default = 4)]
  pub num_workers: usize,

  #[config(default = 42)]
  pub seed: u64,

  #[config(default = 1e-1)]
  pub kld_weight: f32,

  pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
  // Remove existing artifacts before to get an accurate learner summary
  std::fs::remove_dir_all(artifact_dir).ok();
  std::fs::create_dir_all(artifact_dir).ok();
}

/// save tensor into png format
fn save_tensor_as_png<B: burn::tensor::backend::Backend>(
  tensor: &Tensor<B, 3>,
  file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
  // Get the first image from the batch [batch_size, 28, 28] -> [28, 28]
  let first_image: Tensor<B, 2> = tensor.clone().slice([0..1, 0..28, 0..28]).squeeze(0);

  // Convert to CPU backend data for processing
  let data = first_image.to_data();
  let values = data.as_slice::<f32>().unwrap();

  // Convert float values to u8 pixels (assuming values are in range [0.0, 1.0])
  // If your values are in different range (e.g., [-1, 1] or [0, 255]), adjust accordingly
  let pixels: Vec<u8> = values
    .iter()
    .map(|&val| {
      // Clamp to [0.0, 1.0] and convert to [0, 255]
      let clamped = val.max(0.0).min(1.0);
      (clamped * 255.0) as u8
    })
    .collect();

  // Create image buffer (28x28 grayscale)
  let img_buffer = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(28, 28, pixels)
    .ok_or("Failed to create image buffer")?;

  // Save as PNG
  img_buffer.save(file_path)?;

  Ok(())
}

const ARTIFACT_DIR: &str = "vendor";

fn run<B: AutodiffBackend>(device: B::Device) {
  create_artifact_dir(ARTIFACT_DIR);
  create_artifact_dir(format!("{}/{}", ARTIFACT_DIR, "image").as_str());
  let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));

  let config = MnistTrainingConfig::new(config_optimizer);

  B::seed(config.seed);

  let mut vae = VAE::<B>::new(&device, config.kld_weight);
  let batcher = MnistBatcher::default();

  let dataloader_train = DataLoaderBuilder::new(batcher.clone())
    .batch_size(config.batch_size)
    .shuffle(config.seed)
    .num_workers(config.num_workers)
    .build(MnistDataset::train());

  let dataloader_valid = DataLoaderBuilder::new(batcher)
    .batch_size(config.batch_size)
    .shuffle(config.seed)
    .num_workers(config.num_workers)
    .build(MnistDataset::test());

  let train_num_items = dataloader_train.num_items();
  let valid_num_items = dataloader_valid.num_items();

  let mut optimizer = config.optimizer.init();

  for epoch in 1..config.num_epochs + 1 {
    let mut train_loss = 0.0;
    let mut valid_loss = 0.0;

    let mut train_kld_loss = 0.0;
    let mut valid_kld_loss = 0.0;

    let mut train_recon_loss = 0.0;
    let mut valid_recon_loss = 0.0;
    for (_idx, batch) in dataloader_train.iter().enumerate() {
      let output = vae.forward_vae(batch.images.clone());

      train_loss += output.loss.clone().into_scalar().elem::<f32>();
      train_kld_loss += output.kld_loss.into_scalar().elem::<f32>();
      train_recon_loss += output.recon_loss.into_scalar().elem::<f32>();

      let grads = output.loss.backward();
      let grads = GradientsParams::from_grads(grads, &vae);
      vae = optimizer.step(1e-4, vae, grads);
    }
    let avg_train_loss = train_loss / train_num_items as f32;
    let avg_train_kld_loss = train_kld_loss / train_num_items as f32;
    let avg_train_recon_loss = train_recon_loss / train_num_items as f32;

    let valid_vae = vae.valid();

    for (idx, batch) in dataloader_valid.iter().enumerate() {
      let output = valid_vae.forward_vae(batch.images.clone());

      valid_loss += output.loss.clone().into_scalar().elem::<f32>();
      valid_kld_loss += output.kld_loss.into_scalar().elem::<f32>();
      valid_recon_loss += output.recon_loss.into_scalar().elem::<f32>();

      if idx == 0 {
        // save `recon` and `image` into artifact/image, in png format
        let recon_image = output.recon.clone();
        let original_image = batch.images.clone();

        // generate a sample from N(0,I)
        let z = Tensor::random(
          [config.batch_size, 32],
          burn::tensor::Distribution::Normal(0f64, 1f64),
          &device,
        );
        let gen_image = valid_vae.decode(z);
        save_tensor_as_png(
          &recon_image.reshape([config.batch_size, 28, 28]),
          format!("{}/image/{}_recon.png", ARTIFACT_DIR, epoch).as_str(),
        )
        .unwrap();
        save_tensor_as_png(
          &original_image.reshape([config.batch_size, 28, 28]),
          format!("{}/image/{}_original.png", ARTIFACT_DIR, epoch).as_str(),
        )
        .unwrap();
        save_tensor_as_png(
          &gen_image.reshape([config.batch_size, 28, 28]),
          format!("{}/image/{}_gen.png", ARTIFACT_DIR, epoch).as_str(),
        )
        .unwrap();
      }
    }

    let avg_valid_loss = valid_loss / valid_num_items as f32;
    let avg_valid_kld_loss = valid_kld_loss / valid_num_items as f32;
    let avg_valid_recon_loss = valid_recon_loss / valid_num_items as f32;

    [avg_train_loss, avg_train_kld_loss, avg_train_recon_loss]
      .iter()
      .zip(["train/loss", "train/kld_loss", "train/recon_loss"].iter())
      .for_each(|(loss, name)| {
        println!(
          "Epoch {}/{}, {}: {:.2e}",
          epoch, config.num_epochs, name, loss
        );
      });

    [avg_valid_loss, avg_valid_kld_loss, avg_valid_recon_loss]
      .iter()
      .zip(["valid/loss", "valid/kld_loss", "valid/recon_loss"].iter())
      .for_each(|(loss, name)| {
        println!(
          "Epoch {}/{}, {}: {:.2e}",
          epoch, config.num_epochs, name, loss
        );
      });
    if epoch % 10 == 0 || epoch == config.num_epochs {
      valid_vae
        .save_file(
          format!("{ARTIFACT_DIR}/model"),
          &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
    }
  }
}

fn main() {
  use burn::backend::Autodiff;

  run::<Autodiff<Wgpu>>(Default::default());
}
