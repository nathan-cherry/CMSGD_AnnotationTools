# XML Relationship Annotation Tool

This repository contains the software tools developed to facilitate the annotation of the **Coastal Monitoring Scene Graph Dataset (CMSGD)**. These tools were designed to streamline the process of defining semantic and spatial relationships between features in coastal imagery.

## Repository Structure

The repository is organized into two distinct versions of the tool. Both are fully functional for reproducing dataset annotations:

* **`XML Annotation Tool (Original)`**: The legacy version used for the primary relationship annotations during the initial data collection phase of this thesis.
* **`XML Annotation Tool (Current)`**: An updated version featuring a modernized GUI, a scene graph visualizer, and enhanced analytic modules for real-time data insights. 

---

## Getting Started

### Prerequisites
* Python 3.x
* Required dependencies (see `requirements.txt`)
* **Dataset Access:** You will need the XML annotation files and the source imagery. These can be retrieved from the [CMSGD Repository](https://github.com/nathan-cherry/CMSGD.git).

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/nathan-cherry/CMSGD_AnnotationTools.git
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Workflow

The tool follows a linear workflow to ensure data integrity and annotation consistency.

### 1. Loading Data
* **Load XML:** Begin by loading the existing XML annotation file. This file contains the base feature data (masks and bounding boxes).
* **Load Images:** Select the folder containing the corresponding imagery. Once both are linked, the workspace will render the first image.

### 2. Creating Relationships
To annotate a relationship between two entities:
1.  **Source Feature:** Select the primary feature from the feature list. The tool will highlight the corresponding mask/bounding box on the image.
2.  **Target Feature:** Select the second feature. This will be highlighted in a contrasting color to distinguish it from the source.
3.  **Relationship Label:** Enter the relationship type in the label field. 
    * **Consistency:** The tool provides suggestions from `relationship_labels.json` to prevent nomenclature drift.
    * **Automation:** New labels entered are automatically appended to the JSON file for future use.
4.  **Add:** Click the **Add** button to commit the relationship to the current session.

### 3. Saving Progress
Click **Save XML** to write your annotations to the file. You can exit the application and resume at any point by reloading the updated XML file.

---

## Key Features

| Feature | Description |
| :--- | :--- |
| **Dual-Color Highlighting** | Visual feedback for source and target selections to prevent mapping errors. |
| **Label Suggestion Engine** | Minimizes naming inconsistencies by pulling from a centralized JSON registry. |
| **Analytics (Current Version)** | Provides real-time statistics on relationship distribution and annotation density. |
| **Persistence** | Full support for incremental saving and loading, crucial for large-scale research. |



