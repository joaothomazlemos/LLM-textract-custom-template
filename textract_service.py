from collections import defaultdict
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


class TextractKVExtractor:
    def __init__(self, textract_client):
        """
        textract_client: boto3 Textract client (already configured)
        """
        self.client = textract_client

    def _get_kv_map(self, file_bytes):
        response = self.client.analyze_document(
            Document={"Bytes": file_bytes}, FeatureTypes=["FORMS"]
        )
        blocks = response["Blocks"]
        key_map, value_map, block_map = {}, {}, {}

        for block in blocks:
            block_map[block["Id"]] = block
            if block["BlockType"] == "KEY_VALUE_SET":
                if "KEY" in block["EntityTypes"]:
                    key_map[block["Id"]] = block
                else:
                    value_map[block["Id"]] = block
        return key_map, value_map, block_map

    def _find_value_block(self, key_block, value_map):
        for rel in key_block.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for value_id in rel["Ids"]:
                    return value_map.get(value_id)
        return None

    def _get_text(self, block, block_map):
        if not block:
            return ""
        text = ""
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    child = block_map.get(child_id, {})
                    if child.get("BlockType") == "WORD":
                        text += child.get("Text", "") + " "
                    if (
                        child.get("BlockType") == "SELECTION_ELEMENT"
                        and child.get("SelectionStatus") == "SELECTED"
                    ):
                        text += "X "
        return text.strip()

    def _get_kv_relationship(self, key_map, value_map, block_map):
        kvs = defaultdict(list)
        for key_id, key_block in key_map.items():
            value_block = self._find_value_block(key_block, value_map)
            key_text = self._get_text(key_block, block_map)
            value_text = self._get_text(value_block, block_map)
            kvs[key_text].append(value_text)
        return kvs

    def _extract_all_text(self, file_bytes: bytes) -> str:
        """
        Extracts all text from the document, including text not associated with keys.
        """
        response = self.client.detect_document_text(Document={"Bytes": file_bytes})
        text_blocks = [
            block["Text"]
            for block in response["Blocks"]
            if block["BlockType"] == "LINE"
        ]
        return "\n".join(text_blocks)

    def run(self, file_path=None, file_bytes=None, extract_full_text=False):
        """
        Run the extraction.
        Provide either `file_path` or `file_bytes`.
        If extract_full_text=True, returns all text (not just key-value pairs).
        """
        if file_path:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        if not file_bytes:
            raise ValueError("You must provide file_path or file_bytes.")

        if extract_full_text:
            return self._extract_all_text(file_bytes)

        key_map, value_map, block_map = self._get_kv_map(file_bytes)
        kvs = self._get_kv_relationship(key_map, value_map, block_map)
        return str(kvs)


def extract_text_single_id(
    images_bytes_list: List[bytes],
    textract_instance: TextractKVExtractor,
    extract_full_text: bool = False,
) -> str:
    """
    Extract text from a list of PNG image bytes using TextractKVExtractor.
    if extract full text is true, it will not only return key values pairs.
    Args:
        id_and_images: Tuple containing (id, list of PNG image bytes)
        textract_instance: TextractKVExtractor instance

    Returns:
        concatenated extracted text from all images
    """

    full_text = ""
    for img_bytes in images_bytes_list:
        text = textract_instance.run(
            file_bytes=img_bytes, extract_full_text=extract_full_text
        )
        full_text += text
    return full_text


class TextractPDFAnalyzer:
    """
    Analyzes PDF documents using AWS Textract for forms extraction.
    Handles multi-page PDFs through asynchronous processing.
    
    Example:
        # Initialize with your clients
        pdf_analyzer = TextractPDFAnalyzer(
            textract_client=textract_client,
            s3_client=s3_client, 
            bucket_name="your-bucket-name",
            bucket_folder="your-folder-name"
        )

        # Extract forms data from PDF with file identifier
        forms_data = pdf_analyzer.extract_forms_data_from_pdf(
            pdf_bytes=pdf_bytes, 
            file_id="unique_file_identifier"
        )
    """

    def __init__(self, textract_client, s3_client, bucket_name: str, bucket_folder: str):
        """
        Initialize the PDF analyzer.

        Args:
            textract_client: Boto3 Textract client
            s3_client: Boto3 S3 client
            bucket_name: S3 bucket name for temporary PDF storage
            bucket_folder: S3 folder name for organizing PDFs
        """
        self.textract_client = textract_client
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.bucket_folder = bucket_folder
        self.forms_data = {}
        self.tables_data = []
        self.logger = get_logger(__name__)

    def upload_pdf_to_s3(self, pdf_bytes: bytes, key: str) -> bool:
        """
        Upload PDF bytes to S3 bucket.

        Args:
            pdf_bytes: PDF file as bytes
            key: S3 object key

        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=pdf_bytes,
                ContentType="application/pdf",
            )
            self.logger.info(f"PDF uploaded to S3: s3://{self.bucket_name}/{key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload PDF to S3: {e}")
            return False

    def start_pdf_analysis(
        self, s3_key: str, client_request_token: Optional[str] = None
    ) -> str:
        """
        Start asynchronous PDF analysis with Textract.

        Args:
            s3_key: S3 object key where PDF is stored
            client_request_token: Optional idempotency token

        Returns:
            str: JobId for tracking the analysis
        """
        try:
            request_params = {
                "DocumentLocation": {
                    "S3Object": {"Bucket": self.bucket_name, "Name": s3_key}
                },
                "FeatureTypes": ["FORMS", "TABLES"],
            }

            if client_request_token:
                request_params["ClientRequestToken"] = client_request_token

            response = self.textract_client.start_document_analysis(**request_params)
            job_id = response["JobId"]

            self.logger.info(f"Started PDF analysis with JobId: {job_id}")
            return job_id

        except Exception as e:
            self.logger.error(f"Failed to start PDF analysis: {e}")
            raise

    def get_analysis_results(self, job_id: str, max_wait_time: int = 300) -> dict:
        """
        Poll for analysis results until completion.

        Args:
            job_id: JobId from start_pdf_analysis
            max_wait_time: Maximum time to wait in seconds

        Returns:
            dict: Complete analysis results
        """
        import time

        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                response = self.textract_client.get_document_analysis(JobId=job_id)
                job_status = response["JobStatus"]

                if job_status == "SUCCEEDED":
                    self.logger.info(
                        f"PDF analysis completed successfully for JobId: {job_id}"
                    )

                    # Handle pagination to get all results
                    all_blocks = response["Blocks"]
                    next_token = response.get("NextToken")

                    while next_token:
                        response = self.textract_client.get_document_analysis(
                            JobId=job_id, NextToken=next_token
                        )
                        all_blocks.extend(response["Blocks"])
                        next_token = response.get("NextToken")

                    # Return complete response with all blocks
                    response["Blocks"] = all_blocks
                    return response

                elif job_status == "FAILED":
                    self.logger.error(f"PDF analysis failed for JobId: {job_id}")
                    raise Exception(
                        f"Textract analysis failed: {response.get('StatusMessage', 'Unknown error')}"
                    )

                elif job_status == "IN_PROGRESS":
                    self.logger.info(f"PDF analysis in progress for JobId: {job_id}")
                    time.sleep(5)  # Wait 5 seconds before next poll

                else:
                    self.logger.warning(f"Unknown job status: {job_status}")
                    time.sleep(5)

            except Exception as e:
                self.logger.error(f"Error polling for results: {e}")
                raise

        raise TimeoutError(
            f"PDF analysis timed out after {max_wait_time} seconds for JobId: {job_id}"
        )

    def extract_forms_data_from_pdf(
        self, pdf_bytes: bytes, file_id: str, s3_key: Optional[str] = None
    ) -> dict:
        """
        Main method to extract forms data from PDF.

        Args:
            pdf_bytes: PDF file as bytes
            file_id: Unique identifier to include in the uploaded filename
            s3_key: Optional S3 key. If None, generates timestamp-based key

        Returns:
            dict: Extracted forms data
        """
        import uuid
        from datetime import datetime

        try:
            # Generate S3 key if not provided
            if not s3_key:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                s3_key = f"{self.bucket_folder}/{file_id}_{timestamp}_{unique_id}.pdf"

            # Step 1: Upload PDF to S3
            if not self.upload_pdf_to_s3(pdf_bytes, s3_key):
                raise Exception("Failed to upload PDF to S3")

            # Step 2: Start analysis
            client_token = str(uuid.uuid4())
            job_id = self.start_pdf_analysis(s3_key, client_token)

            # Step 3: Get results
            analysis_response = self.get_analysis_results(job_id)

            # Step 4: Extract forms and tables data
            self.forms_data = self._extract_key_value_pairs(analysis_response["Blocks"])
            self.tables_data = self._extract_tables(analysis_response["Blocks"])

            # Combine forms and tables data
            combined_data = {
                "forms": self.forms_data,
                "tables": self.tables_data,
                "summary": {
                    "forms_count": len(self.forms_data),
                    "tables_count": len(self.tables_data)
                }
            }

            # Step 5: Clean up S3 object (optional)
            self._cleanup_s3_object(s3_key)

            self.logger.info(
                f"Successfully extracted data from PDF. Found {len(self.forms_data)} key-value pairs and {len(self.tables_data)} tables"
            )
            return combined_data

        except Exception as e:
            self.logger.error(f"Error in PDF forms extraction: {e}")
            # Attempt cleanup even on error
            if s3_key:
                self._cleanup_s3_object(s3_key)
            raise

    def _extract_key_value_pairs(self, blocks: list) -> dict:
        """
        Extract key-value pairs from Textract blocks.
        Reuses the logic from TextractFormAnalyzer.
        """
        key_map = {}
        value_map = {}
        block_map = {}

        for block in blocks:
            block_id = block["Id"]
            block_map[block_id] = block

            if block["BlockType"] == "KEY_VALUE_SET":
                if "KEY" in block["EntityTypes"]:
                    key_map[block_id] = block
                else:
                    value_map[block_id] = block

        extracted_data = {}

        for key_block_id, key_block in key_map.items():
            if "Relationships" in key_block:
                for relationship in key_block["Relationships"]:
                    if relationship["Type"] == "VALUE":
                        value_block_ids = relationship["Ids"]

                        key_text = self._get_text_from_block(key_block, block_map)
                        value_text = ""

                        for value_block_id in value_block_ids:
                            value_block = value_map.get(value_block_id)
                            if value_block:
                                value_text += self._get_text_from_block(
                                    value_block, block_map
                                )

                        if key_text and value_text:
                            extracted_data[key_text.strip()] = value_text.strip()

        return extracted_data

    def _extract_tables(self, blocks: list) -> list:
        """
        Extract table data from Textract blocks.
        
        Returns:
            list: List of tables, each containing rows with cells
        """
        tables = []
        table_blocks = {}
        cell_blocks = {}
        
        # First pass: organize blocks by type
        for block in blocks:
            if block["BlockType"] == "TABLE":
                table_blocks[block["Id"]] = block
            elif block["BlockType"] == "CELL":
                cell_blocks[block["Id"]] = block
        
        # Process each table
        for table_id, table_block in table_blocks.items():
            # Get all cells for this table
            table_cells = {}
            
            if "Relationships" in table_block:
                for relationship in table_block["Relationships"]:
                    if relationship["Type"] == "CHILD":
                        for cell_id in relationship["Ids"]:
                            if cell_id in cell_blocks:
                                cell_block = cell_blocks[cell_id]
                                row_index = cell_block.get("RowIndex", 0)
                                col_index = cell_block.get("ColumnIndex", 0)
                                
                                # Extract text from cell
                                cell_text = self._get_text_from_block(cell_block, {block["Id"]: block for block in blocks})
                                
                                if row_index not in table_cells:
                                    table_cells[row_index] = {}
                                table_cells[row_index][col_index] = cell_text.strip()
            
            # Convert to structured format
            if table_cells:
                table_data = {
                    "table_id": table_id,
                    "rows": []
                }
                
                # Sort rows and columns
                for row_idx in sorted(table_cells.keys()):
                    row_data = []
                    row_cells = table_cells[row_idx]
                    for col_idx in sorted(row_cells.keys()):
                        row_data.append(row_cells[col_idx])
                    table_data["rows"].append(row_data)
                
                tables.append(table_data)
        
        return tables

    def _get_text_from_block(self, block: dict, block_map: dict) -> str:
        """
        Extract text from a block using relationships.
        Reuses the logic from TextractFormAnalyzer.
        """
        text = ""
        if "Relationships" in block:
            for relationship in block["Relationships"]:
                if relationship["Type"] == "CHILD":
                    for child_id in relationship["Ids"]:
                        child_block = block_map.get(child_id)
                        if child_block and child_block["BlockType"] == "WORD":
                            text += child_block["Text"] + " "
        return text

    def _cleanup_s3_object(self, s3_key: str) -> None:
        """
        Delete the temporary PDF from S3.
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"Cleaned up S3 object: s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup S3 object {s3_key}: {e}")

    def get_forms_data(self) -> dict:
        """Returns the extracted forms data."""
        return self.forms_data

    def get_tables_data(self) -> list:
        """Returns the extracted tables data."""
        return self.tables_data

    def get_all_data(self) -> dict:
        """Returns both forms and tables data in a structured format."""
        return {
            "forms": self.forms_data,
            "tables": self.tables_data,
            "summary": {
                "forms_count": len(self.forms_data),
                "tables_count": len(self.tables_data)
            }
        }
