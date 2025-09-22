"""
signatures.py

Collection of dspy.Signature classes for the LITL IC50 prediction task.
A DSPy signature declaratively defines the behavior of an agent via defining
input and output fields. 
For simplicity, currently only one signature is defined predicting from
a single drug identifier and relevant experimental context.

Classes:
- PredictIC50DrugCell: Predicts the IC50 value for a given drug and cell line,
  along with confidence and explanation.
"""

from typing import Optional

import dspy

class PredictIC50DrugCell(dspy.Signature):
    """
    You are a expert pharmacologist and medicinal chemist, tasked with
    predicting the cell viability IC50 value for a given drug
    against a specific cell line. 
    
    You are given a single drug name that uniquely identifies the drug,
    a single cell line name that uniquely identifies the cell line,
    and optionally an experimental description that provides
    additional context about the assay.

    If available, additional tools can be used to look up more information 
    about the drug target, mechanism of action, etc. If you would like to 
    acquire such information, you MUST explicitly call these tools.
    """
    # for some flexibility to use different kinds of drug identifiers
    drug: str = dspy.InputField(
        desc="The drug name or identifier for which you will predict the IC50")
    cell_line: str = dspy.InputField(
        desc="The cell line name or identifier against which you will "
             "predict the IC50")
    experimental_description: Optional[str] = dspy.InputField(
        desc="Optional description of experimental details that may be "
             "relevant for predicting the IC50, or None if not available")
    output_unit: str = dspy.InputField(
        desc="The unit required for the predicted IC50 value")
    
    ic50_pred: float = dspy.OutputField(
        desc="Your predicted IC50 value (in nM) for the drug against the target"
             ", must be a float value strictly greater than 0"
        )
    confidence: int = dspy.OutputField(
        desc="Your confidence in the IC50 prediction, on a scale of 0-100")
    explanation: str = dspy.OutputField(
        desc="A detailed explanation of how you arrived at your IC50 prediction"
    )