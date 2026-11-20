# from MGCA
import pickle
import numpy as np
import pandas as pd
from medeha.constants import *
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

np.random.seed(0)


# create bounding boxes
def create_bbox(row):
    if row["Target"] == 0:
        return 0
    else:
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["width"]
        y2 = y1 + row["height"]
        return [x1, y1, x2, y2]


def preprocess_rsna_data(test_fac=0.15, seed =0):
    try:
        df = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the RSNA RSNA dataset is \
            stored at {RSNA_DATA_DIR}"
        )

    # class_df = pd.read_csv(RSNA_CLASSINFO_CSV)
    # all_df = pd.merge()

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    # df["Path"] = df["patientId"].apply(
    #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

    # split data
    train_df, test_val_df = train_test_split(
        df, test_size=test_fac * 2, random_state=seed)
    test_df, valid_df = train_test_split(
        test_val_df, test_size=0.5, random_state=seed)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    if os.path.exists(RSNA_TRAIN_CSV):
        os.remove(RSNA_TRAIN_CSV)
    if os.path.exists(RSNA_VALID_CSV):
        os.remove(RSNA_VALID_CSV)
    if os.path.exists(RSNA_TEST_CSV):
        os.remove(RSNA_TEST_CSV)
    train_df.to_csv(RSNA_TRAIN_CSV, index=False)
    valid_df.to_csv(RSNA_VALID_CSV, index=False)
    test_df.to_csv(RSNA_TEST_CSV, index=False)


def prepare_detection_pkl(df, path):
    filenames = []
    bboxs = []
    for row in df.itertuples():
        filename = row.patientId + ".dcm"
        filenames.append(filename)
        if row.Target == 0:
            bboxs.append(np.zeros((1, 4)))
        else:
            y = np.array(row.bbox)
            bboxs.append(y)

    filenames = np.array(filenames)
    bboxs = np.array(bboxs)

    with open(path, "wb") as f:
        pickle.dump([filenames, bboxs], f)


def prepare_detection_data(seed = 0):
    try:
        df = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the RSNA RSNA dataset is \
            stored at {RSNA_DATA_DIR}"
        )

    # class_df = pd.read_csv(RSNA_CLASSINFO_CSV)
    # all_df = pd.merge()

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # split data
    train_df, test_val_df = train_test_split(
        df, test_size=5337 * 2, random_state=seed)
    test_df, valid_df = train_test_split(
        test_val_df, test_size=0.5, random_state=seed)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    if os.path.exists(RSNA_DETECTION_TRAIN_PKL):
        os.remove(RSNA_DETECTION_TRAIN_PKL)
    if os.path.exists(RSNA_DETECTION_VALID_PKL):
        os.remove(RSNA_DETECTION_VALID_PKL)
    if os.path.exists(RSNA_DETECTION_TEST_PKL):
        os.remove(RSNA_DETECTION_TEST_PKL)
    prepare_detection_pkl(
        train_df, RSNA_DETECTION_TRAIN_PKL)
    prepare_detection_pkl(
        valid_df, RSNA_DETECTION_VALID_PKL)
    prepare_detection_pkl(test_df, RSNA_DETECTION_TEST_PKL)



if __name__ == "__main__":
    parser = ArgumentParser(
        "config of rsna dataset maker")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    preprocess_rsna_data(seed =args.seed)
    prepare_detection_data(args.seed)
