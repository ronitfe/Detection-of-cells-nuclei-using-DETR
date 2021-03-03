n_folds = 5
seed = 42
num_classes = 2
num_queries = 50
null_class_coef = 0.5
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 15

IMG_WIDTH = 384
IMG_HEIGHT = 384
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train'
TEST_PATH = 'stage1_test'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def store_bounding_boxes(img, train_id, mask_id, rotby_90):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 2)
    # print(contours)
    cnt = contours[0]
    # print(cnt)
    # print(cv2.boundingRect(cnt)   )
    x, y, w, h = cv2.boundingRect(cnt)

    x = x * (IMG_WIDTH / img.shape[1])
    w = w * (IMG_WIDTH / img.shape[1])
    y = y * (IMG_WIDTH / img.shape[0])
    h = h * (IMG_WIDTH / img.shape[0])

    if (x > IMG_WIDTH - 1):
        x = IMG_WIDTH - 1
    if (y > IMG_HEIGHT - 1):
        y = IMG_HEIGHT - 1
    if (x + w > IMG_WIDTH - 1):
        w = IMG_WIDTH - 1 - x
    if (y + h > IMG_HEIGHT - 1):
        h = IMG_HEIGHT - 1 - y

    bbdict = {"train_id": train_id, "mask_id": mask_id, "rotby_90": rotby_90, "x": x, "y": y,
              "w": w, "h": h}
    return bbdict


def get_train_transforms():
    return A.Compose([
        # A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
        # val_shift_limit=0.2, p=0.9),

        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),

        # A.ToGray(p=0.01),

        # A.HorizontalFlip(p=0.1),

        # A.VerticalFlip(p=0.1),

        A.Resize(height=384, width=384, p=1),

        # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, p=0.5),

        ToTensorV2(p=1.0)],

        # p=1.0,

        # bbox_params=A.BboxParams(format='coco',label_fields=['labels'])
    )


def get_valid_transforms():
    return A.Compose([A.Resize(height=384, width=384, p=1.0),
                      ToTensorV2(p=1.0)],
                     # p=1.0,
                     # bbox_params=A.BboxParams(format='coco',label_fields=['labels'])
                     )