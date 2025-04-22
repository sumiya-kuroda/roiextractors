"""A segmentation extractor for MFH 2p preprocessing pipeline.

Classes
-------
MFHSegmentationExtractor
    A segmentation extractor for MFH 2p preprocessing pipeline.
"""
from pathlib import Path
from warnings import warn

import numpy as np
from ...extraction_tools import PathType
from ...segmentationextractor import SegmentationExtractor
from typing import Optional
from ...extraction_tools import _image_mask_extractor

class MFHSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for 2p-313 ROI segmentation method.

    This class is based on Suite2pSegmentationExtractor, modified for MFH preprocessing pipeline
    """

    extractor_name = "MFHSegmentation"
    is_writable = False
    # mode = "folder"
    installation_mesg = "MFHSegmentationExtractor Failed"  # error message when not installed

    @classmethod
    def get_available_channels(cls, folder_path: PathType) -> list[str]:
        """Get the available channel names from the folder paths produced by Suite2p.

        Parameters
        ----------
        folder_path : PathType
            Path to Suite2p output path.

        Returns
        -------
        channel_names: list
            List of channel names.
        """
        plane_names = cls.get_available_planes(folder_path=folder_path)

        channel_names = ["chan1"]

        return channel_names

    @classmethod
    def get_available_planes(cls, folder_path: PathType) -> list[str]:
        """Get the available plane names from the folder produced by Suite2p.

        Parameters
        ----------
        folder_path : PathType
            Path to Suite2p output path.

        Returns
        -------
        plane_names: list
            List of plane names.
        """
        from natsort import natsorted

        folder_path = Path(folder_path)
        prefix = "plane"
        plane_paths = natsorted(folder_path.glob(pattern=prefix + "*"))
        assert len(plane_paths), f"No planes found in '{folder_path}'."
        plane_names = [plane_path.stem for plane_path in plane_paths]
        return plane_names

    def __init__(
        self,
        folder_path: PathType,
        channel_name: Optional[str] = None,
        plane_name: Optional[str] = None,
        ast_neuropil: Optional[bool] = False
    ):
        """Create SegmentationExtractor object out of suite 2p data type.

        Parameters
        ----------
        folder_path: str or Path
            The path to the 'suite2p' folder.
        channel_name: str, optional
            The name of the channel to load, to determine what channels are available use MFHSegmentationExtractor.get_available_channels(folder_path).
        plane_name: str, optional
            The name of the plane to load, to determine what planes are available use MFHSegmentationExtractor.get_available_planes(folder_path).
        ast_neuropil: bool, optional
            Whether to use ast_neuropil corrected F if exists.
        """

        channel_names = self.get_available_channels(folder_path=folder_path)
        if channel_name is None:
            if len(channel_names) > 1:
                # For backward compatibility maybe it is better to warn first
                warn(
                    "More than one channel is detected! Please specify which channel you wish to load with the `channel_name` argument. "
                    "To see what channels are available, call `MFHSegmentationExtractor.get_available_channels(folder_path=...)`.",
                    UserWarning,
                )
            channel_name = channel_names[0]

        self.channel_name = channel_name
        if self.channel_name not in channel_names:
            raise ValueError(
                f"The selected channel '{channel_name}' is not a valid channel name. To see what channels are available, "
                f"call `MFHSegmentationExtractor.get_available_channels(folder_path=...)`."
            )

        plane_names = self.get_available_planes(folder_path=folder_path)
        if plane_name is None:
            if len(plane_names) > 1:
                # For backward compatibility maybe it is better to warn first
                warn(
                    "More than one plane is detected! Please specify which plane you wish to load with the `plane_name` argument. "
                    "To see what planes are available, call `MFHSegmentationExtractor.get_available_planes(folder_path=...)`.",
                    UserWarning,
                )
            plane_name = plane_names[0]

        if plane_name not in plane_names:
            raise ValueError(
                f"The selected plane '{plane_name}' is not a valid plane name. To see what planes are available, "
                f"call `MFHSegmentationExtractor.get_available_planes(folder_path=...)`."
            )
        self.plane_name = plane_name

        super().__init__()

        self.folder_path = Path(folder_path)

        options = self._load_npy(file_name="ops.npy", require=True)
        self.options = options.item()
        self._sampling_frequency = self.options["fs"]
        self._num_frames = self.options["nframes"]
        self._image_size = (self.options["Ly"], self.options["Lx"])
        if ast_neuropil and not self.options['ast_neuropil']:
            raise ValueError('Ast neuropil correction was not operated')

        self.stat = self._load_npy(file_name="stat.npy", require=True)

        fluorescence_traces_file_name = "F.npy"
        neuropil_traces_file_name = "Fneu.npy"
        if ast_neuropil:
            fluorescence_traces_corrected_file_name = "Fast.npy"
            dff_file_name = "dff_ast.npy"
            deconvolved_file_name = 'spks.npy'
        else:
            fluorescence_traces_corrected_file_name = None
            dff_file_name = "dff.npy"
            deconvolved_file_name = 'spks.npy'

        self._roi_response_raw = self._load_npy(file_name=fluorescence_traces_file_name, mmap_mode="r", transpose=True)
        self._roi_response_neuropil = self._load_npy(file_name=neuropil_traces_file_name, mmap_mode="r", transpose=True)
        self._roi_response_denoised = self._load_npy(file_name=fluorescence_traces_corrected_file_name, mmap_mode="r", transpose=True)
        self._roi_response_dff = self._load_npy(file_name=dff_file_name, mmap_mode="r", transpose=True)
        self._roi_response_deconvolved = (
            self._load_npy(file_name=deconvolved_file_name, mmap_mode="r", transpose=True) if channel_name == "chan1" else None
        )

        self.iscell = self._load_npy("iscell.npy", mmap_mode="r")

        # The name of the OpticalChannel object is "OpticalChannel" if there is only one channel, otherwise it is
        # "Chan1" or "Chan2".
        self._channel_names = ["OpticalChannel" if len(channel_names) == 1 else channel_name.capitalize()]

        # self._image_correlation = self._correlation_image_read()
        # image_mean_name = "meanImg" if channel_name == "chan1" else f"meanImg_chan2"
        # self._image_mean = self.options[image_mean_name] if image_mean_name in self.options else None
        roi_indices = list(range(self.get_num_rois()))
        self._image_masks = _image_mask_extractor(
            self.get_roi_pixel_masks(),
            roi_indices,
            self.get_image_size(),
        )

    def _load_npy(self, file_name: str, mmap_mode=None, transpose: bool = False, require: bool = False):
        """Load a .npy file with specified filename. Returns None if file is missing.

        Parameters
        ----------
        file_name: str
            The name of the .npy file to load.
        mmap_mode: str
            The mode to use for memory mapping. See numpy.load for details.
        transpose: bool, optional
            Whether to transpose the loaded array.
        require: bool, optional
            Whether to raise an error if the file is missing.

        Returns
        -------
            The loaded .npy file.
        """
        if file_name is None:
            if require:
                raise FileNotFoundError("file_name is empty.")
            return    
            
        file_path = self.folder_path / self.plane_name / file_name
        if not file_path.exists():
            if require:
                raise FileNotFoundError(f"File {file_path} not found.")
            return

        data = np.load(file_path, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)
        if transpose:
            return data.T

        return data

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_accepted_list(self) -> list[int]:
        return list(np.where(self.iscell[:, 0] == 1)[0])

    def get_rejected_list(self) -> list[int]:
        return list(np.where(self.iscell[:, 0] == 0)[0])

    def _correlation_image_read(self) -> Optional[np.ndarray]:
        """Read correlation image from ops (settings) dict.

        Returns
        -------
        img : numpy.ndarray | None
            The correlation image.
        """
        if "Vcorr" not in self.options:
            return None

        correlation_image = self.options["Vcorr"]
        if (self.options["yrange"][-1], self.options["xrange"][-1]) == self._image_size:
            return correlation_image

        img = np.zeros(self._image_size, correlation_image.dtype)
        img[
            (self.options["Ly"] - self.options["yrange"][-1]) : (self.options["Ly"] - self.options["yrange"][0]),
            self.options["xrange"][0] : self.options["xrange"][-1],
        ] = correlation_image

        return img

    @property
    def roi_locations(self) -> np.ndarray:
        """Returns the center locations (x, y) of each ROI."""
        return np.array([j["med"] for j in self.stat]).T.astype(int)

    def get_roi_pixel_masks(self, roi_ids=None) -> list[np.ndarray]:
        pixel_mask = []
        for i in range(self.get_num_rois()):
            pixel_mask.append(
                np.vstack(
                    [
                        self.stat[i]["ypix"],
                        self.stat[i]["xpix"],
                        self.stat[i]["lam"],
                    ]
                ).T
            )
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return [pixel_mask[i] for i in roi_idx_]

    def get_image_size(self) -> tuple[int, int]:
        return self._image_size
    