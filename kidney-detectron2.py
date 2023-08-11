import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
#from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
import cv2
import os
import pickle
from pycocotools.coco import COCO

# Prepare Data
data_dir = "kidney_train"
json_file = os.path.join("/home/jaison/gwdg/detectron2", "coco_annotations.json")

# Register the datasetlets
DatasetCatalog.register("kidney_train", lambda: load_coco_json(json_file, data_dir))
MetadataCatalog.get("kidney_train").set(thing_classes=["glomerulus", "blood_vessel", "unsure"])

# Model Configuration
cfg = get_cfg()
cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("kidney_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4  # Adjusted based on GPU memory.
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Three classes: glomerulus, blood_vessel, unsure

# Ensure we're using the GPU
cfg.MODEL.DEVICE = "cuda"

# Train
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.train()

# After training, save predictor
with open(os.path.join(cfg.OUTPUT_DIR, "predictor.pkl"), "wb") as f:
    pickle.dump(trainer.model, f)

# TESTING THE MODEL
# Load the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("kidney_train", cfg, False, output_dir=cfg.OUTPUT_DIR)

# load predictor:
with open(os.path.join(cfg.OUTPUT_DIR, "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

dataset_dicts = load_coco_json(json_file, data_dir)

# Testing loop
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    # Evaluate
    evaluator.reset()
    evaluator.update(outputs["instances"])
    print(evaluator.summarize())

# Get a random image from the dataset
import random

d = random.sample(dataset_dicts, 1)[0]
img = cv2.imread(d["file_name"])

# Visualization
visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("kidney_train"), scale=1.5)
out = visualizer.draw_dataset_dict(d)
cv2.imshow("Ground Truth", out.get_image()[:, :, ::-1])

# Predict
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("kidney_train"), scale=1.5)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"), instance_mode=ColorMode.IMAGE)
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])

cv2.waitKey(0)
