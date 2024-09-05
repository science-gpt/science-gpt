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


# TODO refactor the following function. Define a textextract ABC and then inherit for PDF
class TextExtract:
    # TODO add docstring

    def __init__(self, data_type: RAW_DATA_TYPES):
        """
        Instantiates a TextExtract object.

        :param data_type: type of raw data
        """
        self.data_type = data_type

    def extract(self, data: RawData) -> Text:
        # TODO add docstring
        if self.data_type == "pdf" and isinstance(data, PDFData):
            return self._pdf_extract(data)
        else:
            # TODO implement better logging (maybe integrate Azure monitor)
            raise ValueError(f"Data type {self.data_type} is not supported")

    @staticmethod
    def _pdf_extract(data: PDFData) -> Text:
        """
        Extracts text from a PDF data object

        :param data: the pdf object to be extracted
        :return: text data extracted from the pdf
        """
        # TODO implement text extraction step
        extracted_text = "<place holder>"
        text = Text(text=extracted_text, title=data.name, data_type="pdf")
        return text
