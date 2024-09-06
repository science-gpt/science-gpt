from abc import ABC, abstractmethod
from chunk import Chunk
from typing import List

from raw_data import RAW_DATA_TYPES, PDFData, RawData
from utils import OutputObject


class Text(OutputObject):
    # TODO docstring

    def __init__(self, text: str, title: str, data_type: RAW_DATA_TYPES):
        """
        Instantiates an object of this class.

        :param text: the extracted text
        :param title: a title for the extracted text
        :param data_type: the type of the original data source
        """
        super().__init__(title=title, data_type=data_type)
        self.text = text


class TextExtract(ABC):
    # TODO docstring

    def __init__(self, data_type: RAW_DATA_TYPES):
        """
        Instantiates a TextExtract object.

        :param data_type: type of raw data
        """
        self.data_type = data_type

    @abstractmethod
    def __call__(self, data: RawData) -> List[Text]:
        # TODO docstring
        pass


class PDFExtract(TextExtract):
    # TODO docstring

    def __init__(self, data_type: RAW_DATA_TYPES):
        # TODO docstring
        super().__init__(data_type=data_type)
        assert (
            data_type == "pdf"
        )  # TODO there should be a better way to validate this, pydantic?

    def __call__(self, data: PDFData) -> List[Text]:
        # TODO docstring
        # TODO add pdf extraction code
        pass
