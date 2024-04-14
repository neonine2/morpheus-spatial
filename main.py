from lightning.pytorch import seed_everything

seed_everything(42)
import morpheus as mp

data_path = "/groups/mthomson/zwang2/IMC/output/cedarsLiver_sz48_pxl3_nc44/temp/singlecell.csv"  # change to your own directory
dataset = mp.SpatialDataset(input_path=data_path)

patch_size = 16  # Patch size in pixels
pixel_size = 3  # Pixel size in microns
cell_types = ["Tcytotoxic", "Tumor"]  # Specify the cell types of interest
mask_cell_types = ["Tcytotoxic"]
dataset.generate_masked_patch(
    cell_to_mask=mask_cell_types,
    cell_types=cell_types,
    patch_size=patch_size,
    pixel_size=pixel_size,
    save=True,
)

# example metadata
print(dataset.metadata.head())


colname = "Contains_Tcytotoxic"
dataset.generate_data_splits(stratify_by=colname)

# # initialize model
# model_arch = "unet"
# n_channels = dataset.n_channels
# img_size = dataset.img_size
# model = mp.PatchClassifier(n_channels, img_size, model_arch)

# # train model
# trainer_params = {
#     "max_epochs": 100,  # set to >60 for better performance
#     "accelerator": "auto",
#     "logger": False,
# }
# model = mp.train(
#     model=model,
#     dataset=dataset,
#     label_name=colname,
#     trainer_params=trainer_params,
# )

# images to generate counterfactuals
dataset.get_split_info()
select_metadata = dataset.metadata[
    (dataset.metadata["Contains_Tumor"] == 1)
    & (dataset.metadata["Contains_Tcytotoxic"] == 0)
    & (dataset.metadata["splits"] == "train")
]

# channels allowed to be perturbed
channel_to_perturb = [
    "Glnsynthetase",
    "CCR4",
    "PDL1",
    "LAG3",
    "CD105endoglin",
    "TIM3",
    "CXCR4",
    "PD1",
    "CYR61",
    "CD44",
    "IL10",
    "CXCL12",
    "CXCR3",
    "Galectin9",
    "YAP",
]

# probability cutoff for classification
threshold = 0.43

# optimization parameters
optimization_param = {
    "use_kdtree": True,
    "theta": 40.0,
    "kappa": (threshold - 0.5) * 2,
    "learning_rate_init": 0.1,
    "beta": 1.0,
    "max_iterations": 1000,  # set to >1000 for better performance
    "c_init": 1000.0,
    "c_steps": 5,
}

# load model
model = mp.load_model(f"{dataset.root_dir}/model/checkpoints/epoch=41-step=12432.ckpt")


# Generate counterfactuals using trained model
cf = mp.get_counterfactual(
    images=select_metadata,
    dataset=dataset,
    target_class=1,
    model=model,
    channel_to_perturb=channel_to_perturb,
    optimization_params=optimization_param,
    threshold=threshold,
    save_dir=f"{dataset.root_dir}/cf/",
    device="cpu",
    num_workers=64,
)
