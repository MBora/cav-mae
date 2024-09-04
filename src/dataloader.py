# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row["index"]] = row["display_name"]
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class VGGSound(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        four_tiles=False,
        restrict_randomness=False,
    ):
        if four_tiles == True:
            print("now use 4 tiles")
        else:
            print("now use 1 tile")
        if restrict_randomness == True:
            print("now restrict randomness")
        else:
            print("normal random")
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        self.four_tiles = four_tiles
        self.restrict_randomness = restrict_randomness

        self.data = data_json["data"]
        self.data = self.pro_data(self.data)
        print("Dataset has {:d} samples".format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        print("Using Label Smoothing: " + str(self.label_smooth))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm", 0)
        self.timem = self.audio_conf.get("timem", 0)
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup", 0)
        print("now using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process " + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        if self.skip_norm:
            print(
                "now skip normalization (use it ONLY when you are computing the normalization stats)."
            )
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(
                    self.norm_mean, self.norm_std
                )
            )

        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise", False)
        if self.noise == True:
            print("now use noise augmentation")
        else:
            print("not use noise augmentation")

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))

        self.target_length = self.audio_conf.get("target_length")

        # train or eval
        self.mode = self.audio_conf.get("mode")
        print("now in {:s} mode.".format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get("frame_use", -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get("total_frame", 10)
        print(
            "now use frame {:d} from total {:d} frames".format(
                self.frame_use, self.total_frame
            )
        )

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get("im_res", 224)
        print("now using {:d} * {:d} image input".format(self.im_res, self.im_res))
        if self.four_tiles:
            self.preprocess = T.Compose(
                [
                    T.Resize([112, 112], interpolation=PIL.Image.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )
        else:
            self.preprocess = T.Compose(
                [
                    T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                    T.CenterCrop(self.im_res),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )

        self.audio_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/audio_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.video_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/video_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.audio_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )
        self.video_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )

        self.epoch_id = -1
        self.next_epoch()

    def shuffle_audio_order(self):
        random.shuffle(self.audio_order)

    def shuffle_video_order(self):
        random.shuffle(self.video_order)

    def next_epoch(self):
        self.epoch_id += 1
        if self.restrict_randomness:
            if self.epoch_id % 4 == 0:
                self.shuffle_audio_order()
            if self.epoch_id % 40 == 0:
                self.shuffle_video_order()

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]["wav"],
                data_json[i]["labels"],
                data_json[i]["video_id"],
                data_json[i]["video_path"],
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum["wav"] = np_data[0]
        datum["labels"] = np_data[1]
        datum["video_id"] = np_data[2]
        datum["video_path"] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10,
            )
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            # print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_img(self, video_id, video_path):
        if self.mode == "eval":
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = self.epoch_id % 10

        while (
            os.path.exists(
                video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"
            )
            == False
            and frame_idx >= 1
        ):
            # print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"
        # print(out_path)
        return out_path

    def get_4_images(self, video_id, video_path):
        x = list(range(10))
        random.shuffle(x)
        tensor = torch.zeros([3, 224, 224])
        for i in range(4):
            filename = video_path + "/frame_" + str(x[i]) + "/" + video_id + ".jpg"
            img = Image.open(filename)
            img = self.preprocess(img)
            tensor[
                :,
                112 * (i % 2) : 112 * (i % 2) + 112,
                112 * (i // 2) : 112 * (i // 2) + 112,
            ] = img
        return tensor

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum["wav"], mix_datum["wav"], mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["video_id"], datum["video_path"])
                else:
                    image = self.get_image(
                        self.randselect_img(datum["video_id"], datum["video_path"]),
                        self.randselect_img(mix_datum["video_id"], datum["video_path"]),
                        mix_lambda,
                    )
            except:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (
                    1.0 - self.label_smooth
                )
            for label_str in mix_datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (
                    1.0 - self.label_smooth
                )
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            try:
                fbank = self._wav2fbank(datum["wav"], None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["video_id"], datum["video_path"])
                else:
                    image = self.get_image(
                        self.randselect_img(datum["video_id"], datum["video_path"]),
                        None,
                        0,
                    )
            except:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(
                fbank, np.random.randint(-self.target_length, self.target_length), 0
            )

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        if self.restrict_randomness:

            audio_mask = self.audio_file[:, self.audio_order[index], :]
            video_mask = self.video_file[:, self.video_order[index], :]

            return fbank, image, label_indices, audio_mask, video_mask

        else:

            return fbank, image, label_indices

    def __len__(self):
        return self.num_samples


class AudiosetDataset(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        four_tiles=False,
        restrict_randomness=False,
    ):
        if four_tiles == True:
            print("now use 4 tiles")
        else:
            print("now use 1 tile")
        if restrict_randomness == True:
            print("now restrict randomness")
        else:
            print("normal random")
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        self.four_tiles = four_tiles
        self.restrict_randomness = restrict_randomness

        self.data = data_json["data"]
        self.data = self.pro_data(self.data)
        print("Dataset has {:d} samples".format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        print("Using Label Smoothing: " + str(self.label_smooth))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm", 0)
        self.timem = self.audio_conf.get("timem", 0)
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup", 0)
        print("now using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process " + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        if self.skip_norm:
            print(
                "now skip normalization (use it ONLY when you are computing the normalization stats)."
            )
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(
                    self.norm_mean, self.norm_std
                )
            )

        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise", False)
        if self.noise == True:
            print("now use noise augmentation")
        else:
            print("not use noise augmentation")

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))

        self.target_length = self.audio_conf.get("target_length")

        # train or eval
        self.mode = self.audio_conf.get("mode")
        print("now in {:s} mode.".format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get("frame_use", -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get("total_frame", 10)
        print(
            "now use frame {:d} from total {:d} frames".format(
                self.frame_use, self.total_frame
            )
        )

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get("im_res", 224)
        print("now using {:d} * {:d} image input".format(self.im_res, self.im_res))
        if self.four_tiles:
            self.preprocess = T.Compose(
                [
                    T.Resize([112, 112], interpolation=PIL.Image.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )
        else:
            self.preprocess = T.Compose(
                [
                    T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                    T.CenterCrop(self.im_res),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )

        self.audio_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/audio_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.video_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/video_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.audio_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )
        self.video_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )

        self.epoch_id = -1
        self.next_epoch()

    def shuffle_audio_order(self):
        random.shuffle(self.audio_order)

    def shuffle_video_order(self):
        random.shuffle(self.video_order)

    def next_epoch(self):
        self.epoch_id += 1
        if self.restrict_randomness:
            if self.epoch_id % 4 == 0:
                self.shuffle_audio_order()
            if self.epoch_id % 40 == 0:
                self.shuffle_video_order()

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]["wav"],
                data_json[i]["labels"],
                data_json[i]["video_id"],
                data_json[i]["video_path"],
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum["wav"] = np_data[0]
        datum["labels"] = np_data[1]
        datum["video_id"] = np_data[2]
        datum["video_path"] = np_data[3]
        return datum

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10,
            )
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            # print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def randselect_img(self, video_id, video_path):
        if self.mode == 'eval':
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)

        while os.path.exists(video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg') == False and frame_idx >= 1:
            print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg'
        return out_path

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum["wav"], mix_datum["wav"], mix_lambda)
            except:
                raise FileNotFoundError(f"{datum['wav']} or {mix_datum['wav']} does not exist.")
                # fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["video_id"], datum["video_path"])
                else:
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), self.randselect_img(mix_datum['video_id'], datum['video_path']), mix_lambda)
            except:
                raise FileNotFoundError(f"{datum['video_path']}/*/{datum['video_id']} or {mix_datum['video_path']}/*/{mix_datum['video_id']} does not exist.")
                # image = torch.zeros([3,224,224])
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (
                    1.0 - self.label_smooth
                )
            for label_str in mix_datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (
                    1.0 - self.label_smooth
                )
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            try:
                fbank = self._wav2fbank(datum["wav"], None, 0)
            except:
                raise FileNotFoundError(f"{datum['wav']} does not exist.")
                # fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["video_id"], datum["video_path"])
                else:
                    image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), None, 0)
            except:
                raise FileNotFoundError(f"{datum['video_path']}/*/{datum['video_id']}.jpg does not exist.")
                # image = torch.zeros([3,224,224])
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(
                fbank, np.random.randint(-self.target_length, self.target_length), 0
            )

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        if self.restrict_randomness:

            audio_mask = self.audio_file[:, self.audio_order[index], :]
            video_mask = self.video_file[:, self.video_order[index], :]

            return fbank, image, label_indices, audio_mask, video_mask

        else:

            return fbank, image, label_indices

    def __len__(self):
        return self.num_samples

class KineticsDataset(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        four_tiles=False,
        restrict_randomness=False,
    ):
        if four_tiles == True:
            print("now use 4 tiles")
        else:
            print("now use 1 tile")
        if restrict_randomness == True:
            print("now restrict randomness")
        else:
            print("normal random")
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        self.four_tiles = four_tiles
        self.restrict_randomness = restrict_randomness

        self.data = data_json["data"]
        self.data = self.pro_data(self.data)
        print("Dataset has {:d} samples".format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        print("Using Label Smoothing: " + str(self.label_smooth))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm", 0)
        self.timem = self.audio_conf.get("timem", 0)
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup", 0)
        print("now using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process " + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        if self.skip_norm:
            print(
                "now skip normalization (use it ONLY when you are computing the normalization stats)."
            )
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(
                    self.norm_mean, self.norm_std
                )
            )

        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise", False)
        if self.noise == True:
            print("now use noise augmentation")
        else:
            print("not use noise augmentation")

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))

        self.target_length = self.audio_conf.get("target_length")

        # train or eval
        self.mode = self.audio_conf.get("mode")
        print("now in {:s} mode.".format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get("frame_use", -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get("total_frame", 10)
        print(
            "now use frame {:d} from total {:d} frames".format(
                self.frame_use, self.total_frame
            )
        )

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get("im_res", 224)
        print("now using {:d} * {:d} image input".format(self.im_res, self.im_res))
        if self.four_tiles:
            self.preprocess = T.Compose(
                [
                    T.Resize([112, 112], interpolation=PIL.Image.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )
        else:
            self.preprocess = T.Compose(
                [
                    T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                    T.CenterCrop(self.im_res),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )

        self.audio_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/audio_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.video_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/video_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.audio_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )
        self.video_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )

        self.epoch_id = -1
        self.next_epoch()

    def shuffle_audio_order(self):
        random.shuffle(self.audio_order)

    def shuffle_video_order(self):
        random.shuffle(self.video_order)

    def next_epoch(self):
        self.epoch_id += 1
        if self.restrict_randomness:
            if self.epoch_id % 4 == 0:
                self.shuffle_audio_order()
            if self.epoch_id % 40 == 0:
                self.shuffle_video_order()

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]["audio"],
                data_json[i]["label"],
                data_json[i]["id"],
                data_json[i]["video_path"],
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum["audio"] = np_data[0]
        datum["label"] = np_data[1]
        datum["id"] = np_data[2]
        datum["video_path"] = np_data[3]
        return datum

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10,
            )
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            # print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_frame(self, video_path, mix_path=None, mix_lambda=0.):
        if mix_path == None:
            img = np.load(video_path)[random.randint(0,9)]
            img = torch.from_numpy(img).permute(2,0,1) / 255.
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = np.load(video_path)[random.randint(0,9)]
            img1 = torch.from_numpy(img1).permute(2,0,1) / 255.
            image_tensor1 = self.preprocess(img1)

            img2 = np.load(mix_path)[random.randint(0,9)]
            img2 = torch.from_numpy(img2).permute(2,0,1) / 255.
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1. - mix_lambda) * image_tensor2
            return image_tensor

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum["audio"], mix_datum["audio"], mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["id"], datum["video_path"])
                else:
                    image = self.randselect_frame(
                        datum["video_path"],
                        mix_datum["video_path"],
                        mix_lambda,
                    )
            except:
                raise FileNotFoundError(f"{datum['video_path']} or {mix_datum['video_path']} does not exist.")
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            for label_str in datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (
                    1.0 - self.label_smooth
                )
            for label_str in mix_datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (
                    1.0 - self.label_smooth
                )
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            try:
                fbank = self._wav2fbank(datum["audio"], None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["id"], datum["video_path"])
                else:
                    image = self.randselect_frame(datum["video_path"])
            except:
                raise FileNotFoundError(f"{datum['video_path']} does not exist.")
            for label_str in datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(
                fbank, np.random.randint(-self.target_length, self.target_length), 0
            )

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        if self.restrict_randomness:

            audio_mask = self.audio_file[:, self.audio_order[index], :]
            video_mask = self.video_file[:, self.video_order[index], :]

            return fbank, image, label_indices, audio_mask, video_mask

        else:

            return fbank, image, label_indices

    def __len__(self):
        return self.num_samples


class ImageNetDataset(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        four_tiles=False,
        restrict_randomness=False,
    ):
        if four_tiles == True:
            print("now use 4 tiles")
        else:
            print("now use 1 tile")
        if restrict_randomness == True:
            print("now restrict randomness")
        else:
            print("normal random")
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        self.four_tiles = four_tiles
        self.restrict_randomness = restrict_randomness

        self.data = data_json["data"]
        self.data = self.pro_data(self.data)
        print("Dataset has {:d} samples".format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        print("Using Label Smoothing: " + str(self.label_smooth))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm", 0)
        self.timem = self.audio_conf.get("timem", 0)
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup", 0)
        print("now using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process " + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        if self.skip_norm:
            print(
                "now skip normalization (use it ONLY when you are computing the normalization stats)."
            )
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(
                    self.norm_mean, self.norm_std
                )
            )

        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise", False)
        if self.noise == True:
            print("now use noise augmentation")
        else:
            print("not use noise augmentation")

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))

        self.target_length = self.audio_conf.get("target_length")

        # train or eval
        self.mode = self.audio_conf.get("mode")
        print("now in {:s} mode.".format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get("frame_use", -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get("total_frame", 10)
        print(
            "now use frame {:d} from total {:d} frames".format(
                self.frame_use, self.total_frame
            )
        )

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get("im_res", 224)
        print("now using {:d} * {:d} image input".format(self.im_res, self.im_res))
        if self.four_tiles:
            self.preprocess = T.Compose(
                [
                    T.Resize([112, 112], interpolation=PIL.Image.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )
        else:
            self.preprocess = T.Compose(
                [
                    T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                    T.CenterCrop(self.im_res),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )

        self.audio_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/audio_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.video_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/video_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.audio_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )
        self.video_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )

        self.epoch_id = -1
        self.next_epoch()

    def shuffle_audio_order(self):
        random.shuffle(self.audio_order)

    def shuffle_video_order(self):
        random.shuffle(self.video_order)

    def next_epoch(self):
        self.epoch_id += 1
        if self.restrict_randomness:
            if self.epoch_id % 4 == 0:
                self.shuffle_audio_order()
            if self.epoch_id % 40 == 0:
                self.shuffle_video_order()

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]["label"],
                data_json[i]["id"],
                data_json[i]["video_path"],
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum["label"] = np_data[0]
        datum["id"] = np_data[1]
        datum["video_path"] = np_data[2]
        return datum

    def select_image(self, image_path, mix_path=None, mix_lambda=0.):
        if mix_path == None:
            img = Image.open(image_path).convert('RGB')
            img = self.preprocess(img)
            return img
        else:
            img1 = Image.open(image_path).convert('RGB')
            img1 = self.preprocess(img1)

            img2 = Image.open(mix_path).convert('RGB')
            img2 = self.preprocess(img2)

            image_tensor = mix_lambda * img1 + (1. - mix_lambda) * img2
            return image_tensor

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            # try:
            #     fbank = self._wav2fbank(datum["audio"], mix_datum["audio"], mix_lambda)
            # except:
            #     fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["id"], datum["video_path"])
                else:
                    image = self.select_image(
                        datum["video_path"].replace("/mnt/MIG_Store/Datasets/imagenet", ""),
                        mix_datum["video_path"].replace("", ""),
                        mix_lambda,
                    )
            except:
                raise FileNotFoundError(f"{datum['video_path']} or {mix_datum['video_path']} does not exist.")
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            for label_str in datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (
                    1.0 - self.label_smooth
                )
            for label_str in mix_datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (
                    1.0 - self.label_smooth
                )
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            # try:
            #     fbank = self._wav2fbank(datum["audio"], None, 0)
            # except:
            #     fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["id"], datum["video_path"])
                else:
                    image = self.select_image(datum["video_path"])
            except:
                raise FileNotFoundError(f"{datum['video_path']} does not exist.")
            for label_str in datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # fbank = fbank.unsqueeze(0)
        # if self.freqm != 0:
        #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # fbank = fbank.squeeze(0)
        # fbank = torch.transpose(fbank, 0, 1)

        # # normalize the input for both training and test
        # if self.skip_norm == False:
        #     fbank = (fbank - self.norm_mean) / (self.norm_std)
        # # skip normalization the input ONLY when you are trying to get the normalization stats.
        # else:
        #     pass

        # if self.noise == True:
        #     fbank = (
        #         fbank
        #         + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        #     )
        #     fbank = torch.roll(
        #         fbank, np.random.randint(-self.target_length, self.target_length), 0
        #     )

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        if self.restrict_randomness:

            audio_mask = self.audio_file[:, self.audio_order[index], :]
            video_mask = self.video_file[:, self.video_order[index], :]

            return torch.Tensor([0.]), image, label_indices, audio_mask, video_mask

        else:

            return torch.Tensor([0.]), image, label_indices

    def __len__(self):
        return self.num_samples


class SSv2Dataset(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        four_tiles=False,
        restrict_randomness=False,
    ):
        if four_tiles == True:
            print("now use 4 tiles")
        else:
            print("now use 1 tile")
        if restrict_randomness == True:
            print("now restrict randomness")
        else:
            print("normal random")
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)

        self.four_tiles = four_tiles
        self.restrict_randomness = restrict_randomness

        self.data = data_json["data"]
        self.data = self.pro_data(self.data)
        print("Dataset has {:d} samples".format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        print("Using Label Smoothing: " + str(self.label_smooth))
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm", 0)
        self.timem = self.audio_conf.get("timem", 0)
        print(
            "now using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup", 0)
        print("now using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process " + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        if self.skip_norm:
            print(
                "now skip normalization (use it ONLY when you are computing the normalization stats)."
            )
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input.".format(
                    self.norm_mean, self.norm_std
                )
            )

        # if add noise for data augmentation
        self.noise = self.audio_conf.get("noise", False)
        if self.noise == True:
            print("now use noise augmentation")
        else:
            print("not use noise augmentation")

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))

        self.target_length = self.audio_conf.get("target_length")

        # train or eval
        self.mode = self.audio_conf.get("mode")
        print("now in {:s} mode.".format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get("frame_use", -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get("total_frame", 10)
        print(
            "now use frame {:d} from total {:d} frames".format(
                self.frame_use, self.total_frame
            )
        )

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get("im_res", 224)
        print("now using {:d} * {:d} image input".format(self.im_res, self.im_res))
        if self.four_tiles:
            self.preprocess = T.Compose(
                [
                    T.Resize([112, 112], interpolation=PIL.Image.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )
        else:
            self.preprocess = T.Compose(
                [
                    T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                    T.CenterCrop(self.im_res),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            )

        self.audio_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/audio_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.video_file = (
            torch.from_numpy(np.load("/home/amukherjee/Static/video_keeps_60.npy"))[:4]
            if self.restrict_randomness
            else None
        )
        self.audio_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )
        self.video_order = (
            list(range(self.__len__())) if self.restrict_randomness else []
        )

        self.epoch_id = -1
        self.next_epoch()

    def shuffle_audio_order(self):
        random.shuffle(self.audio_order)

    def shuffle_video_order(self):
        random.shuffle(self.video_order)

    def next_epoch(self):
        self.epoch_id += 1
        if self.restrict_randomness:
            if self.epoch_id % 4 == 0:
                self.shuffle_audio_order()
            if self.epoch_id % 40 == 0:
                self.shuffle_video_order()

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                data_json[i]["label"],
                data_json[i]["id"],
                data_json[i]["video_path"],
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum["label"] = np_data[0]
        datum["id"] = np_data[1]
        datum["video_path"] = np_data[2]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10,
            )
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            # print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_img(self, video_id, video_path):
        if self.mode == "eval":
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = self.epoch_id % 10

        while (
            os.path.exists(
                video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"
            )
            == False
            and frame_idx >= 1
        ):
            # print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + "/frame_" + str(frame_idx) + "/" + video_id + ".jpg"
        # print(out_path)
        return out_path

    def get_4_images(self, video_id, video_path):
        x = list(range(10))
        random.shuffle(x)
        tensor = torch.zeros([3, 224, 224])
        for i in range(4):
            filename = video_path + "/frame_" + str(x[i]) + "/" + video_id + ".jpg"
            img = Image.open(filename)
            img = self.preprocess(img)
            tensor[
                :,
                112 * (i % 2) : 112 * (i % 2) + 112,
                112 * (i // 2) : 112 * (i // 2) + 112,
            ] = img
        return tensor

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            # try:
            #     fbank = self._wav2fbank(datum["wav"], mix_datum["wav"], mix_lambda)
            # except:
            #     fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["id"], datum["video_path"])
                else:
                    image = self.get_image(
                        self.randselect_img(datum["id"], datum["video_path"]),
                        self.randselect_img(mix_datum["id"], datum["video_path"]),
                        mix_lambda,
                    )
            except:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            for label_str in datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (
                    1.0 - self.label_smooth
                )
            for label_str in mix_datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (
                    1.0 - self.label_smooth
                )
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (
                self.label_smooth / self.label_num
            )
            # try:
            #     fbank = self._wav2fbank(datum["wav"], None, 0)
            # except:
            #     fbank = torch.zeros([self.target_length, 128]) + 0.01
            try:
                if self.four_tiles:
                    image = self.get_4_images(datum["id"], datum["video_path"])
                else:
                    image = self.get_image(
                        self.randselect_img(datum["id"], datum["video_path"]),
                        None,
                        0,
                    )
            except:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            for label_str in datum["label"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # fbank = fbank.unsqueeze(0)
        # if self.freqm != 0:
        #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # fbank = fbank.squeeze(0)
        # fbank = torch.transpose(fbank, 0, 1)

        # # normalize the input for both training and test
        # if self.skip_norm == False:
        #     fbank = (fbank - self.norm_mean) / (self.norm_std)
        # # skip normalization the input ONLY when you are trying to get the normalization stats.
        # else:
        #     pass

        # if self.noise == True:
        #     fbank = (
        #         fbank
        #         + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        #     )
        #     fbank = torch.roll(
        #         fbank, np.random.randint(-self.target_length, self.target_length), 0
        #     )

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        if self.restrict_randomness:

            audio_mask = self.audio_file[:, self.audio_order[index], :]
            video_mask = self.video_file[:, self.video_order[index], :]

            return torch.Tensor([0.]), image, label_indices, audio_mask, video_mask

        else:

            return torch.Tensor([0.]), image, label_indices

    def __len__(self):
        return self.num_samples
