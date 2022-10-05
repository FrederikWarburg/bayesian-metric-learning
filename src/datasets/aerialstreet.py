import os

import pandas as pd
import numpy as np
import torch.utils.data as data
import torch
from sklearn.neighbors import NearestNeighbors

from datasets.datahelpers import default_loader, imresize, angle_diff


class BaseDataset(data.Dataset):
    def __init__(
        self,
        name,
        mode="train",
        envs=["nordhavn"],  # ,"lolland", "skagen","motorring_3"],
        imsize=None,
        transform=None,
        loader=default_loader,
        posDistThr=10,
        negDistThr=25,
        root_dir="data",
    ):

        self.dbImages = []
        self.qImages = []
        self.qidxs = []
        self.pidxs = []
        self.clusters = []

        # hyper-parameters
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.imsize = imsize

        self.transform = transform
        self.loader = loader

        # flags
        self.envs = envs
        self.name = name
        # TODO: create train and test set
        self.mode = ""  #'test' if mode in ('test', 'val') else 'train'

        # other
        self.transform = transform

        # load query / database data
        for env in self.envs:
            env_path = f"{root_dir}/{env}"
            for seq in os.listdir(f"{env_path}/street_images_perspective"):
                print(f"==> env : {env} {seq}")
                # TODO: something funcky is going on when we also iterate over both model and seq. I think things are added model times...
                # get len of images from cities so far for indexing
                _lenQ = len(self.qImages)
                _lenDb = len(self.dbImages)

                # load query data
                q_path = f"{env_path}/street_images_perspective/{seq}"
                qData = pd.read_csv(f"{q_path}/reference_perspective.csv")

                # append image keys with full path
                self.qImages.extend(
                    [
                        f"{q_path}/images/{k}.jpg"
                        for k in qData["perspective_filename"].values
                    ]
                )
                utmQ = qData[["UTM Easting", "UTM Northing"]].values.reshape(-1, 2)
                casQ = qData["perspective_heading[deg]"].values

                # load database data
                db_path = f"{env_path}/synthetized_view"
                utmDb, casDb = [], []
                for model in os.listdir(db_path):
                    dbData = pd.read_csv(f"{db_path}/{model}/{seq}/reference.csv")

                    # append image keys with full path
                    self.dbImages.extend(
                        [
                            f"{db_path}/{model}/{seq}/images/{k}"
                            for k in dbData["rgb"].values
                        ]
                    )
                    utmDb = (
                        np.concatenate(
                            [
                                utmDb,
                                dbData[["easting", "northing"]].values.reshape(-1, 2),
                            ],
                            axis=0,
                        )
                        if len(utmDb) > 0
                        else dbData[["easting", "northing"]].values.reshape(-1, 2)
                    )
                    casDb = (
                        np.concatenate([casDb, dbData["angle"].values], axis=0)
                        if len(casDb) > 0
                        else dbData["angle"].values
                    )

                # find positive images for training
                neigh = NearestNeighbors(algorithm="brute")
                neigh.fit(utmDb)
                _, pI = neigh.radius_neighbors(utmQ, self.posDistThr)

                if self.mode == "train":
                    _, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

                for qidx in range(len(qData)):

                    # the query image has at least one positive
                    if len(pI[qidx]) > 0:

                        self.qidxs.append(qidx + _lenQ)
                        self.pidxs.append(
                            [
                                p + _lenDb
                                for p in pI[qidx]
                                if abs(angle_diff(casQ[qidx], casDb[p])) < 22.5
                            ]
                        )

                        # in training we have two thresholds, one for finding positives and one for finding images that we are certain are negatives.
                        if self.mode == "train":
                            self.clusters.append([n + _lenDb for n in nI[qidx]])

                self.utmDb = (
                    np.concatenate([self.utmDb, utmDb], axis=0)
                    if hasattr(self, "utmDb")
                    else utmDb
                )
                self.casDb = (
                    np.concatenate([self.casDb, casDb], axis=0)
                    if hasattr(self, "casDb")
                    else casDb
                )
                self.utmQ = (
                    np.concatenate([self.utmQ, utmQ], axis=0)
                    if hasattr(self, "utmQ")
                    else utmQ
                )
                self.casQ = (
                    np.concatenate([self.casQ, casQ], axis=0)
                    if hasattr(self, "casQ")
                    else casQ
                )

        # cast to np.arrays for indexing during training
        self.qImages = np.asarray(self.qImages)
        self.dbImages = np.asarray(self.dbImages)
        self.qidxs = np.asarray(self.qidxs)
        self.pidxs = np.asarray(self.pidxs, dtype=object)
        self.clusters = np.asarray(self.clusters, dtype=object)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):

        raise NotImplementedError

    def __repr__(self):
        fmt_str = self.__class__.__name__ + "\n"
        fmt_str += "    Name and mode: {} {}\n".format(self.name, self.mode)
        fmt_str += "    Number of query images: {}\n".format(len(self.qImages))
        fmt_str += "    Number of database images: {}\n".format(len(self.dbImages))
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class TrainDataset(BaseDataset):
    def __init__(
        self,
        name,
        mode,
        envs=["nordhavn", "lolland", "skagen", "motorring_3"],
        imsize=None,
        transform=None,
        loader=default_loader,
        posDistThr=10,
        negDistThr=25,
        root_dir="data",
    ):
        super().__init__(
            name,
            mode,
            envs,
            imsize,
            transform,
            loader,
            posDistThr,
            negDistThr,
            root_dir,
        )

    def __len__(self):

        return len(self.qidxs)

    def __getitem__(self, index):

        qidx = self.qidxs[index]
        pidx = self.pidxs[index]

        pidx = np.random.choice(pidx, 1)[0]

        qpath, utmQ = self.qImages[qidx], self.utmQ[qidx]
        ppath, utmDb = self.dbImages[pidx], self.utmDb[pidx]

        output = []
        output.append(self.loader(qpath))
        output.append(self.loader(ppath))

        target = []
        target.append(utmQ)
        target.append(utmDb)

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]

        if self.transform is not None:
            output = [self.transform(output[i]) for i in range(len(output))]

        return output, target


class TestDataset(BaseDataset):
    def __init__(
        self,
        name,
        mode,
        envs=["nordhavn", "lolland", "skagen", "motorring_3"],
        imsize=None,
        transform=None,
        loader=default_loader,
        posDistThr=10,
        negDistThr=25,
        root_dir="data",
    ):
        super().__init__(
            name,
            mode,
            envs,
            imsize,
            transform,
            loader,
            posDistThr,
            negDistThr,
            root_dir,
        )

    def __len__(self):
        # the dataset is the queries followed by the database images
        return len(self.qidxs) + len(self.dbImages)

    def __getitem__(self, index):

        if index < len(self.qidxs):
            path = self.qImages[self.qidxs[index]]
            utm = self.utmQ[self.qidxs[index]]
            index = [index, -1]
        else:
            path = self.dbImages[index - len(self.qidxs)]
            utm = self.utmDb[index - len(self.qidxs)]
            index = [-1, index - len(self.qidxs)]

        img = self.loader(path)

        if self.imsize is not None:
            img = imresize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        return img, index, utm
