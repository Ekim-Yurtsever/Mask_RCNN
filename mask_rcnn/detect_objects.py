import os
import sys
import skimage.io
import model as modellib
import visualize
import coco
import numpy as np
import matplotlib.pyplot as plt

def filter_masks(masks, rois, class_ids, scores):
    # cyan 0,255,255
    N = rois.shape[0]
    if N == 0:
        return masks, rois, class_ids, scores
    cropped_regions = np.array([[410, 480, 0, 692],
                               [50, 120, 170, 570],
                                [0, 20, 0, 692]])
    indices = np.ones(N, dtype=bool)
    for i in range(N):
        # Filter classes
        if class_ids[i] == 3: # 3 is car
            indices[i] = True
        elif class_ids[i] == 4: # motorcycle
            indices[i] = True
        elif class_ids[i] == 6: # bus
            indices[i] = True
        elif class_ids[i] == 8: # truck
            indices[i] = True
        else:
            indices[i] = False
            continue
        mask = masks[:, :, i]
        mask_sum = 0
        for region in cropped_regions:
            mask_region=mask[region[0]:region[1], region[2]:region[3]]
            mask_sum += np.sum(mask_region)
        if mask_sum > 0:
            indices[i] = False
    return masks[:, :, indices], rois[indices, :], class_ids[indices], scores[indices]

# Import COCO config
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = ROOT_DIR + '/mask_rcnn_coco.h5'

# Directory of images to run detection on
# IMAGE_DIR = ROOT_DIR + '/raw_images/'
IMAGE_DIR = "/home/kusanagi/Documents/datasets/risk_dataset/lane_change_images/"
OUTPUT_DIR = ROOT_DIR + '/processed_images/'

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG..', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

foldernames = [f for f in os.listdir(IMAGE_DIR) if f.isnumeric() and not f.startswith('.')]
foldernames.sort()
for foldername in foldernames:
    print(foldername)
    if 1: # used to filter folders
        CURRENT_IMAGE_DIR = IMAGE_DIR + foldername + '/'
        CURRENT_OUTPUT_DIR = OUTPUT_DIR + foldername + '/'
        if not os.path.isdir(CURRENT_OUTPUT_DIR):
            os.mkdir(CURRENT_OUTPUT_DIR)
        filenames = sorted(os.listdir(CURRENT_IMAGE_DIR))
        for filename in filenames:
            print(filename)
            if not filename.startswith('.'):
                img = skimage.io.imread(CURRENT_IMAGE_DIR + filename)
                results = model.detect([img], verbose=1)
                # Visualize results
                r = results[0]
                r['masks'], r['rois'], r['class_ids'], r['scores'] = filter_masks(r['masks'], r['rois'], r['class_ids'], r['scores'])
                visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], save_option=1, save_path=CURRENT_OUTPUT_DIR + filename)
                # generate the video from the terminal with the followiung command (ffmpeg has to be installed on the system)
                #ffmpeg -f image2 -r 24 -pattern_type glob -i '*.jpg' -vcodec mpeg4 -y movie11.mp4
                plt.close('all')

#################
# TO GET AN INTERMEDIATE LAYER
# from keras import backend as K
#
# # with a Sequential model
# get_nth_layer_output = K.function([model.keras_model.layers[0].input],
#                                   [model.keras_model.layers[343].output])
#
# #model.keras_model.layers[343].output this is the 5th level resnet conv output
#
# #model.keras_model.layers[359].get_output_at(0)
#
# molded_images, image_metas, windows = model.mold_inputs([image])
# layer_output = get_nth_layer_output([molded_images, image_metas])[0]

#################

# # Run detection
# results = model.detect([image], verbose=1)
# # Visualize results
# r = results[0]
# print(r['masks'])
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], save_option=1,
#                             save_path=os.listdir(ROOT_DIR + '/processed_images/' + img_dir) + filename)
