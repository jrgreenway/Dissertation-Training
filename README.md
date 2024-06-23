# The Plan

### Step 1: Data Collection and Preparation

Collect AIS Data

    Sources: Use maritime databases, simulation tools, or historical collision data.
    Data Points: Ensure each entry includes:
        MMSI (Maritime Mobile Service Identity)
        Latitude and Longitude
        Speed Over Ground (SOG)
        Course Over Ground (COG)
        Heading
        Navigation Status
        Timestamp

Obtain COLREGs Text

    Sources: Access the official COLREGs documentation.
    Segmentation: Segment the text into individual rules for easier processing.

Label Data

    Scenario Creation: Create scenarios using AIS data and label them with the relevant COLREGs rules that apply to those scenarios.

### Step 2: Data Formatting

Format AIS Data and COLREGs Rules

    AIS Data Conversion: Convert AIS data for each scenario into a descriptive text format.
        Example: "Vessel A: MMSI 123456789 at latitude 52.205 and longitude 0.1218 with speed over ground 12.5 and course over ground 85.0. Heading is 90 and navigation status is underway. Vessel B: MMSI 987654321 at latitude 52.205 and longitude 0.1250 with speed over ground 10.0 and course over ground 270.0. Heading is 270 and navigation status is underway."
    Labeling: Pair each descriptive text with the relevant COLREGs rule(s).

Create Training Dataset

    JSON Format: Structure the dataset in JSON format.
        Example: Each scenario includes AIS data for two vessels, the labeled COLREGs rule, and the text of the rule.

### Step 3: Model Training

Pre-Training with COLREGs Text

    Install Necessary Tools: Ensure you have the required libraries and tools.
    Load COLREGs Text: Load the text of the COLREGs rules and prepare it for pre-training the model.
        Tokenizer: Use a tokenizer suitable for BERT.
        Text Segmentation: Segment the COLREGs text into smaller parts if necessary.
    Pre-Training: Pre-train the model on the COLREGs text to ensure it understands the rules.

Fine-Tuning with AIS Scenarios

    Prepare Dataset: Load and preprocess the dataset containing AIS scenarios and the corresponding COLREGs rules.
        Tokenization: Tokenize the AIS scenario descriptions and COLREGs text.
        Dataset Preparation: Format the data appropriately for training.
    Fine-Tuning: Fine-tune the pre-trained model with the AIS scenario descriptions and the COLREGs rules, teaching the model the application context.

### Step 4: Scenario Testing

    Input New AIS Data: Develop a mechanism to input new AIS data for two vessels.
    Convert to Text: Convert the new AIS data into the same narrative text format used during training.
    Use Fine-Tuned Model: Input the converted text into the fine-tuned model to predict the relevant COLREGs rules.

### Step 5: Model Evaluation

    Evaluate Outputs: Compare the model’s predictions with documented outcomes and expert interpretations from established scenarios.
    Define Metrics: Use metrics such as accuracy, precision, recall, and F1-score to assess the model’s performance.
    Refine Model: Based on the evaluation results, refine your model and data preprocessing steps to improve accuracy and reliability.

Example Workflow

    Data Collection:
        Collect AIS data for various scenarios.
        Collect COLREGs text.

    Data Preprocessing:
        Convert AIS data into descriptive text.
        Pair descriptive texts with relevant COLREGs rules.

    Model Training:
        Pre-train the model on the COLREGs text.
        Fine-tune the model using the AIS scenario descriptions and COLREGs rules.

    Scenario Testing:
        Input new AIS data.
        Use the model to predict relevant COLREGs rules.

    Evaluation and Refinement:
        Compare predictions to known outcomes.
        Refine the model based on performance metrics.

## Motivations

- An easily traceable tool to ascertain which rules need to be applied for any given collision scenario.
