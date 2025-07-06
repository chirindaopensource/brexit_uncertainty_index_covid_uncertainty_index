# README.md

# Brexit-Related Uncertainty Index (BRUI)

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-150458.svg?style=flat&logo=python&logoColor=white)](https://www.statsmodels.org/stable/index.html)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white)](https://spacy.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.02439-b31b1b.svg)](https://arxiv.org/abs/2507.02439)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2507.02439-blue)](https://doi.org/10.48550/arXiv.2507.02439)
[![Research](https://img.shields.io/badge/Research-Quantitative%20Finance-green)](https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index)
[![Discipline](https://img.shields.io/badge/Discipline-Econometrics-blue)](https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index)
[![Methodology](https://img.shields.io/badge/Methodology-NLP%20%26%20VAR-orange)](https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index)
[![Text Processing](https://img.shields.io/badge/Text-Processing-blue)](https://spacy.io/)
[![Time Series](https://img.shields.io/badge/Time%20Series-Analysis-red)](https://www.statsmodels.org/stable/index.html)
[![Data Source](https://img.shields.io/badge/Data%20Source-EIU%20Reports-lightgrey)](https://www.eiu.com/)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index)

**Repository:** https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent** implementation of the research methodology from the 2025 paper entitled **"Introducing a New Brexit-Related Uncertainty Index: Its Evolution and Economic Consequences"** by:

*   Ismet Gocer
*   Julia Darby
*   Serdar Ongan

The project provides a robust, end-to-end Python pipeline for constructing a high-frequency, quantifiable measure of geopolitical risk, specifically focusing on "Brexit uncertainty." It transforms this abstract concept into a tangible, decision-useful metric and further analyzes its macroeconomic impacts using Vector Autoregression (VAR) models.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_brexit_uncertainty_analysis](#key-callable-run_brexit_uncertainty_analysis)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Introducing a New Brexit-Related Uncertainty Index: Its Evolution and Economic Consequences." The core of this repository is the iPython Notebook `brexit_related_uncertainty_index_draft.ipynb`, which contains a comprehensive suite of functions to construct the Brexit-Related Uncertainty Index (BRUI) and its complementary COVID-19 Related Uncertainty Index (CRUI), and to analyze their economic consequences.

Measuring geopolitical risk is a critical challenge in modern finance and economics. Abstract concepts like "uncertainty" must be transformed into quantifiable metrics to be useful for risk pricing, capital allocation, and policy formulation. This framework provides a rigorous, data-driven approach to this problem.

This codebase enables researchers, policymakers, and financial analysts to:
-   Construct a high-frequency, text-based measure of Brexit uncertainty from raw documents.
-   Methodologically disentangle Brexit-related uncertainty from concurrent shocks like the COVID-19 pandemic.
-   Analyze the dynamic impact of uncertainty shocks on key macroeconomic variables.
-   Replicate and extend the findings of the original research paper.

## Theoretical Background

The implemented methods are grounded in a combination of advanced Natural Language Processing (NLP) and standard time-series econometrics:

**Context-Aware Uncertainty Attribution:** The core of the index construction is a novel algorithm that moves beyond simple keyword counting. It requires the co-occurrence of an "uncertainty" term and a "Brexit" term within a small, defined text window (10 words on either side). This ensures that the measured uncertainty is contextually relevant.

**Proportional Allocation for Disentanglement:** In periods where both Brexit and COVID-19 are discussed together, the algorithm does not discard the data. Instead, it uses a proportional allocation mechanism based on the relative frequency of "pure" Brexit and "pure" COVID-19 uncertainty mentions in the same document to disentangle the two effects.

**Vector Autoregression (VAR) Modeling:** To assess the economic impact of the newly constructed index, the pipeline employs a standard VAR model. This multivariate time-series model captures the dynamic interdependencies between the BRUI and key macroeconomic variables (GDP, CPI, trade, etc.).

**Cholesky Decomposition for Identification:** To identify the causal impact of an uncertainty shock, a Cholesky decomposition is applied to the VAR model's residuals. By ordering the BRUI first in the system, the model operates under the standard economic assumption that uncertainty shocks are contemporaneously exogenous—they affect the economy within the same month, but are not themselves affected by the economy within that same month.

## Features

The provided iPython Notebook (`brexit_related_uncertainty_index_draft.ipynb`) implements the full research pipeline, including:

-   **Parameter Validation:** Rigorous checks for all input data and configurations to ensure methodological compliance.
-   **Data Cleansing:** Robust handling of missing and non-finite values, and precise temporal filtering.
-   **Advanced NLP Pipeline:** Text normalization, tokenization, context-aware stopword removal, and n-gram analysis.
-   **LLM-Powered Entity Recognition:** Use of SpaCy's `en_core_web_lg` to prepare for entity-based analysis.
-   **Context-Aware Attribution Algorithm:** The core algorithm for identifying and classifying uncertainty mentions.
-   **Index Construction:** Proportional allocation, standardization, and max-normalization to create the final BRUI and CRUI.
-   **Econometric Data Preparation:** Systematic stationarity testing (ADF) and data transformations (log, differencing).
-   **VAR Modeling:** Automated optimal lag selection, model estimation, and diagnostic testing.
-   **Post-Estimation Analysis:** Calculation of Impulse Response Functions (IRFs), Forecast Error Variance Decompositions (FEVDs), and bootstrapped confidence intervals.
-   **Publication-Quality Visualization:** A suite of functions to generate the key figures from the paper.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Text Corpus Processing (Steps 1-8):** The pipeline ingests monthly EIU reports, cleans the text, and processes it using a standard NLP workflow (normalization, tokenization, stopword removal).
2.  **Context-Aware Attribution (Step 9):** For each "uncertainty" keyword, a 21-word context window is analyzed. The presence of Brexit and/or COVID-19 keywords within this window determines the classification of the uncertainty mention.
3.  **Proportional Allocation (Step 10):** For "mixed" contexts containing both Brexit and COVID-19 keywords, the uncertainty count is allocated proportionally to the BRUI and CRUI based on the relative prevalence of "pure" mentions in the same document.
4.  **Index Finalization (Step 11):** The raw uncertainty counts are standardized by the total word count of the report and then normalized so that the maximum value of the index over the entire sample period is 100.
5.  **Econometric Analysis:** The final BRUI is integrated into a VAR model with key UK macroeconomic variables. The model is used to generate IRFs that trace the economic impact of a one-standard-deviation shock to the BRUI.

## Core Components (Notebook Structure)

The `brexit_related_uncertainty_index_draft.ipynb` notebook is structured as a logical pipeline with modular functions for each task:

-   **Task 0: `validate_parameters`**: The initial quality gate for all inputs.
-   **Task 1: `cleanse_data`**: Handles data quality and temporal scoping.
-   **Task 2: `prepare_lexicons`**: Optimizes keyword lists for high-performance matching.
-   **Task 3: `preprocess_text_corpus`**: The foundational NLP pipeline.
-   **Task 4: `load_spacy_model_for_ner`, `apply_ner_to_corpus`**: LLM-based entity extraction.
-   **Task 5: `attribute_uncertainty_in_corpus`**: The core uncertainty attribution algorithm.
-   **Task 6 & 7: `calculate_brui`, `calculate_crui`**: Final index construction.
-   **Task 8: `prepare_data_for_var`**: Prepares data for econometric modeling.
-   **Task 9: `estimate_var_model`**: Estimates and identifies the VAR model.
-   **Task 10: `run_post_estimation_analysis`**: Computes IRFs, FEVDs, and confidence intervals.
-   **Task 11: `plot_...` functions**: The visualization suite.
-   **Main Orchestrator: `run_brexit_uncertainty_analysis`**: Executes the entire pipeline.

## Key Callable: run_brexit_uncertainty_analysis

The central function in this project is `run_brexit_uncertainty_analysis`. It orchestrates the entire analytical workflow from raw data to final results.

```python
def run_brexit_uncertainty_analysis(
    df_input: pd.DataFrame,
    uncertainty_lexicon: List[str],
    brexit_lexicon: List[str],
    covid_lexicon: List[str],
    index_construction_config: Dict[str, Any],
    econometric_analysis_config: Dict[str, Any],
    brexit_events_for_plotting: Dict[str, str],
    comparison_indices_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Executes the end-to-end research pipeline for the Brexit Uncertainty Index.

    This orchestrator function serves as the master controller for the entire
    analysis. It sequentially executes all tasks from parameter validation to
    final visualization, ensuring a rigorous, reproducible, and auditable
    workflow. Each step's outputs are programmatically passed to the next, and
    all significant results, data, and logs are compiled into a comprehensive
    final dictionary.

    Args:
        df_input (pd.DataFrame): The raw input DataFrame containing monthly
            macroeconomic data and the EIU text corpus.
        uncertainty_lexicon (List[str]): The raw list of uncertainty keywords.
        brexit_lexicon (List[str]): The raw list of Brexit-related keywords.
        covid_lexicon (List[str]): The raw list of COVID-19 related keywords.
        index_construction_config (Dict[str, Any]): Configuration for the
            text-based index construction.
        econometric_analysis_config (Dict[str, Any]): Configuration for the
            econometric VAR analysis.
        brexit_events_for_plotting (Dict[str, str]): A dictionary of key Brexit
            events for annotating the final BRUI time-series plot.
        comparison_indices_df (pd.DataFrame, optional): A DataFrame containing
            alternative indices (e.g., BRUI_B, BRUI_C) for validation plotting.
            Must have a DatetimeIndex. Defaults to None.

    Returns:
        Dict[str, Any]: A comprehensive dictionary containing all results.
    """
    # ... (implementation)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `scipy`, `nltk`, `spacy`.
-   NLTK data packages: `punkt`, `stopwords`.
-   SpaCy model: `en_core_web_lg`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index.git
    cd brexit_uncertainty_index_covid_uncertainty_index
    ```

2.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy statsmodels matplotlib scipy nltk spacy
    ```

3.  **Download required NLP data:**
    ```sh
    python -m nltk.downloader punkt stopwords
    python -m spacy download en_core_web_lg
    ```

## Input Data Structure

The primary input is a `pandas.DataFrame` with the following structure:
-   **Index:** A `DatetimeIndex` with monthly frequency ('MS'), covering the full sample period (e.g., '2012-05-01' to '2025-01-01').
-   **Columns:**
    -   `BRUI`: A placeholder column (e.g., filled with zeros).
    -   `GDP`, `CPI`, `PPI`, `X`, `M`, `GBP_EUR`, `GBP_USD`, `EMP`, `UEMP`: Numeric columns containing the macroeconomic data.
    -   `EIU`: An object/string column containing the full text of the monthly EIU report.

See the usage example for a template of how to construct this DataFrame.

## Usage

1.  **Prepare Inputs:** Construct the input DataFrame, lexicons, and configuration dictionaries as shown in the detailed usage example within the `brexit_related_uncertainty_index_draft.ipynb` notebook.
2.  **Open and Run Notebook:** Open the notebook in a Jupyter environment.
3.  **Execute All Cells:** Run all cells in the notebook to define the functions and prepare the example data.
4.  **Invoke the Orchestrator:** The final cells of the notebook demonstrate how to call the main `run_brexit_uncertainty_analysis` function with all the prepared inputs.
5.  **Analyze Outputs:** The returned dictionary will contain all results, logs, and figures generated by the pipeline.

## Output Structure

The `run_brexit_uncertainty_analysis` function returns a single, comprehensive dictionary with the following top-level keys:

-   `audit_logs`: Contains detailed logs from the data cleansing, VAR data preparation, and VAR estimation steps.
-   `final_data`: Contains key data artifacts, including the `prepared_lexicons` object, the final DataFrame with the `BRUI`, `CRUI`, and all intermediate calculation columns, and the stationary dataset used for the VAR.
-   `fitted_model`: Contains the `statsmodels.VARResults` object, which is the complete fitted VAR model.
-   `analysis_results`: Contains the results of the post-estimation analysis, including the IRF point estimates, confidence intervals, and FEVD summary tables.
-   `visualizations`: Contains the `matplotlib.figure.Figure` objects for the generated plots.

## Project Structure

```
brexit_uncertainty_index_covid_uncertainty_index/
│
├── brexit_related_uncertainty_index_draft.ipynb  # Main implementation notebook
├── LICENSE                                       # MIT license file
└── README.md                                     # This documentation file
```

## Customization

The pipeline is highly customizable via the `index_construction_config` and `econometric_analysis_config` dictionaries. Users can easily modify:
-   **Lexicons:** Add or remove keywords from the input lists.
-   **Context Window Size:** Change `context_window_size` in the configuration.
-   **VAR Model:** Add or remove variables, change the lag selection criterion, or modify the Cholesky ordering in the `econometric_analysis_config`.
-   **Post-Estimation:** Adjust the `horizon` for IRFs/FEVDs or the `confidence_interval_level`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{gocer2025introducing,
  title={Introducing a New Brexit-Related Uncertainty Index: Its Evolution and Economic Consequences},
  author={Gocer, Ismet and Darby, Julia and Ongan, Serdar},
  journal={arXiv preprint arXiv:2507.02439},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of the Brexit-Related Uncertainty Index (BRUI). 
GitHub repository: https://github.com/chirindaopensource/brexit_uncertainty_index_covid_uncertainty_index
```

## Acknowledgments

-   Credit to Ismet Gocer, Julia Darby, and Serdar Ongan for their novel methodology in constructing a context-aware uncertainty index.
-   Thanks to the developers of the `statsmodels`, `pandas`, `spacy`, and `nltk` libraries, which are the foundational pillars of this analytical pipeline.
