# <center> **ETL** - PDF to Markdown Conversion

This notebook demonstrates how to use two ETL frameworks, **PyMuPDF** and **Docling**, to efficiently convert PDF documents into Markdown files. It supports processing multiple PDFs, saving the output as structured Markdown files for use in downstream tasks such as document indexing or text analysis.  


## Setup and Installation 
Ensure you have the required dependencies installed:  
```
!pip install pymupdf pymupdf4llm docling -q
```

## Usage  
1. **Enter project details:**  
   - When prompted, provide the project name and the topic name. The notebook will organize input and output files under structured directories:
     - Input PDFs: `projects/{project}/pdfs/{topic}`
     - Output Markdown:  
       - PyMuPDF: `projects/{project}/mds_pymupdf/{topic}`  
       - Docling: `projects/{project}/mds_docling/{topic}`  

2. **PDF Conversion:**  
   - Each framework processes all PDF files in the specified input directory and saves the converted Markdown files to the corresponding output directory.  

### Frameworks Used
#### 1. **[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)**  
   - **Function:** Converts PDFs to Markdown with PyMuPDF’s `to_markdown` function.  
   - **Implementation:**  
     ```python
     def pymupdf_pdf_to_md(input_pdf_path, output_md_path):
         # Open PDF and convert specified pages to Markdown
         # Save output to a file
     ```

#### 2. **[Docling](https://github.com/DS4SD/docling)**  
   - **Function:** Converts PDFs to Markdown using Docling’s `DocumentConverter`.  
   - **Implementation:**  
     ```python
     def docling_pdf_to_md(input_pdf_path, output_md_path):
         # Convert PDF to Markdown using Docling
         # Save output to a file
     ```

## Directory Structure
The input and output directories follow a logical structure based on the provided project and topic names:  
```
projects/
└── {project}/
    ├── pdfs/
    │   └── {topic}/      # Input PDF files
    ├── mds_pymupdf/
    │   └── {topic}/      # Markdown files from PyMuPDF
    └── mds_docling/
        └── {topic}/      # Markdown files from Docling
```

## Execution Workflow  
1. **PyMuPDF Conversion**  
   - Iterates through all `.pdf` files in the input directory, converts them to Markdown using `to_markdown`, and saves the output.

2. **Docling Conversion**  
   - Iterates through all `.pdf` files, converts them using Docling's `DocumentConverter`, and saves the output as Markdown.

Both methods handle exceptions to avoid disruptions during batch processing.

## Output  
- Converted Markdown files are saved with the same base name as the input PDF, but with an `.md` extension.  
- Example: `example.pdf` → `example.md`

**Conversion Status:**  
For each file, a message is displayed indicating its processing status. Upon completion, the notebook outputs:  
```
----- Conversion completed! -----
```  