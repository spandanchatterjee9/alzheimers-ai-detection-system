"""
Script to generate the complete, submission-ready academic project synopsis in .odt format:
'Alzheimers_Dementia_Stage_Classification_Synopsis.odt'
using odfpy library.
"""

import os
import sys
from pathlib import Path

from odf.opendocument import OpenDocumentText
from odf.style import (
    Style, TextProperties, ParagraphProperties, TableColumnProperties,
    TableCellProperties, PageLayout, PageLayoutProperties, MasterPage,
    Header, Footer
)
from odf.text import P, H, List, ListItem, Span
from odf.table import Table, TableColumn, TableRow, TableCell

def create_synopsis_odt(output_filename="Alzheimers_Dementia_Stage_Classification_Synopsis.odt"):
    doc = OpenDocumentText()

    # Page Break Style
    pb_style = Style(name="PageBreakStyle", family="paragraph")
    pb_style.addElement(ParagraphProperties(breakbefore="page"))
    doc.styles.addElement(pb_style)

    # --- STYLE DEFINITIONS ---
    # Page Layout (A4, 2cm margins)
    pl = PageLayout(name="StandardPageLayout")
    pl.addElement(PageLayoutProperties(
        pagewidth="21.0cm",
        pageheight="29.7cm",
        margintop="2.0cm",
        marginbottom="2.0cm",
        marginleft="2.5cm",
        marginright="2.5cm"
    ))
    doc.automaticstyles.addElement(pl)

    master = MasterPage(name="Standard", pagelayoutname="StandardPageLayout")
    doc.masterstyles.addElement(master)

    # Styles
    # Title Style
    title_style = Style(name="DocTitle", family="paragraph")
    title_style.addElement(ParagraphProperties(alignment="center", spacebefore="1.5cm", spaceafter="0.5cm"))
    title_style.addElement(TextProperties(fontname="Arial", fontsize="22pt", fontweight="bold", color="#1e3a8a"))
    doc.styles.addElement(title_style)

    # Subtitle Style
    subtitle_style = Style(name="DocSubtitle", family="paragraph")
    subtitle_style.addElement(ParagraphProperties(alignment="center", spacebefore="0.2cm", spaceafter="1.5cm"))
    subtitle_style.addElement(TextProperties(fontname="Arial", fontsize="14pt", fontitalic="italic", color="#3b82f6"))
    doc.styles.addElement(subtitle_style)

    # Meta Block Style
    meta_style = Style(name="DocMeta", family="paragraph")
    meta_style.addElement(ParagraphProperties(alignment="center", spacebefore="0.5cm", spaceafter="0.3cm"))
    meta_style.addElement(TextProperties(fontname="Arial", fontsize="11pt", color="#334155"))
    doc.styles.addElement(meta_style)

    # Heading 1 Style
    h1_style = Style(name="Heading1", family="paragraph")
    h1_style.addElement(ParagraphProperties(spacebefore="0.8cm", spaceafter="0.3cm", keepwithnext="always"))
    h1_style.addElement(TextProperties(fontname="Arial", fontsize="16pt", fontweight="bold", color="#1e3a8a"))
    doc.styles.addElement(h1_style)

    # Heading 2 Style
    h2_style = Style(name="Heading2", family="paragraph")
    h2_style.addElement(ParagraphProperties(spacebefore="0.5cm", spaceafter="0.2cm", keepwithnext="always"))
    h2_style.addElement(TextProperties(fontname="Arial", fontsize="13pt", fontweight="bold", color="#0f766e"))
    doc.styles.addElement(h2_style)

    # Heading 3 Style
    h3_style = Style(name="Heading3", family="paragraph")
    h3_style.addElement(ParagraphProperties(spacebefore="0.3cm", spaceafter="0.1cm", keepwithnext="always"))
    h3_style.addElement(TextProperties(fontname="Arial", fontsize="11pt", fontweight="bold", color="#334155"))
    doc.styles.addElement(h3_style)

    # Body Text Style
    body_style = Style(name="BodyText", family="paragraph")
    body_style.addElement(ParagraphProperties(alignment="justify", lineheight="115%", spaceafter="0.25cm"))
    body_style.addElement(TextProperties(fontname="Times New Roman", fontsize="11pt", color="#1e293b"))
    doc.styles.addElement(body_style)

    # Callout / Alert Box Style
    callout_style = Style(name="CalloutText", family="paragraph")
    callout_style.addElement(ParagraphProperties(alignment="left", spacebefore="0.3cm", spaceafter="0.3cm", marginleft="0.5cm", marginright="0.5cm"))
    callout_style.addElement(TextProperties(fontname="Times New Roman", fontsize="10.5pt", fontitalic="italic", color="#1e3a8a"))
    doc.styles.addElement(callout_style)

    # Table Header Style
    th_style = Style(name="TableHeader", family="paragraph")
    th_style.addElement(ParagraphProperties(alignment="center"))
    th_style.addElement(TextProperties(fontname="Arial", fontsize="10pt", fontweight="bold", color="#ffffff"))
    doc.styles.addElement(th_style)

    # Table Cell Style
    td_style = Style(name="TableCell", family="paragraph")
    td_style.addElement(ParagraphProperties(alignment="left"))
    td_style.addElement(TextProperties(fontname="Times New Roman", fontsize="10pt", color="#1e293b"))
    doc.styles.addElement(td_style)

    # Bold inline text
    bold_span = Style(name="BoldText", family="text")
    bold_span.addElement(TextProperties(fontweight="bold"))
    doc.styles.addElement(bold_span)

    # Helper Functions
    def add_h1(text):
        doc.text.addElement(H(outlinelevel=1, stylename=h1_style, text=text))

    def add_h2(text):
        doc.text.addElement(H(outlinelevel=2, stylename=h2_style, text=text))

    def add_h3(text):
        doc.text.addElement(H(outlinelevel=3, stylename=h3_style, text=text))

    def add_p(text):
        p = P(stylename=body_style)
        p.addText(text)
        doc.text.addElement(p)

    def add_bullet(text, bold_prefix=""):
        p = P(stylename=body_style)
        if bold_prefix:
            s = Span(stylename=bold_span)
            s.addText(bold_prefix + " ")
            p.addElement(s)
        p.addText(text)
        doc.text.addElement(p)

    # --- TITLE PAGE ---
    doc.text.addElement(P(stylename=title_style, text="Explainable Deep Learning for Alzheimer's-Related Dementia Stage Classification Using Structural MRI"))
    doc.text.addElement(P(stylename=subtitle_style, text="An OASIS-1 Based Medical Imaging and Web-Based Decision Support System"))
    
    doc.text.addElement(P(stylename=meta_style, text="ACADEMIC PROJECT SYNOPSIS"))
    doc.text.addElement(P(stylename=meta_style, text="Degree: Bachelor of Technology / Computer Science & Engineering"))
    doc.text.addElement(P(stylename=meta_style, text="Domain: Medical Imaging AI, Deep Learning, & Clinical Decision Support"))
    doc.text.addElement(P(stylename=meta_style, text="Dataset Benchmark: OASIS-1 Cross-Sectional MRI Cohort"))
    doc.text.addElement(P(stylename=meta_style, text="Academic Year: 2025–2026"))

    doc.text.addElement(P(stylename=pb_style))

    # --- SECTION 1: ABSTRACT ---
    add_h1("1. Abstract")
    add_p(
        "Alzheimer's disease (AD) represents the primary cause of neurodegenerative dementia globally, characterized "
        "by progressive synaptic disruption, neuronal loss, and brain structural atrophy. Early discrimination of dementia "
        "severity stages is paramount for timely therapeutic intervention and clinical trial stratification. This project "
        "presents an explainable, 3D volumetric deep-learning framework for four-class Alzheimer's-related dementia stage "
        "classification based on structural Magnetic Resonance Imaging (sMRI) scans from the Open Access Series of Imaging "
        "Studies (OASIS-1) cross-sectional cohort. The primary objective is to classify sMRI volumes into Clinical Dementia "
        "Rating (CDR) severity stages: Cognitively Normal (CDR 0), Very Mild Dementia (CDR 0.5), Mild Dementia (CDR 1), "
        "and Moderate Dementia (CDR 2)."
    )
    add_p(
        "The system incorporates a robust, 13-stage modular preprocessing engine that transforms raw volumetric Analyze 7.5 "
        "scans into canonical RAS-oriented, intensity-normalized (Z-score), brain-cropped 3D tensors (128 × 128 × 128) and "
        "quality-filtered 2D planar slices (224 × 224). To eliminate data leakage, a strict stratified patient-wise splitting "
        "protocol is enforced across the 234 labelled OASIS-1 subjects (80% Train / 10% Validation / 10% Test), ensuring "
        "complete subject isolation across partitions. A 3D ResNet-18 neural network serves as the primary deep-learning "
        "architecture, capturing spatial 3D anatomical relationships across the cerebral cortex and medial temporal structures."
    )
    add_p(
        "To provide interpretable insights for clinicians, Gradient-Weighted Class Activation Mapping (Grad-CAM / Grad-CAM++) "
        "is integrated to visualize salient anatomical regions driving model predictions. An overall dementia-related classification "
        "probability is derived by aggregating non-zero CDR class probabilities. Finally, the preprocessing engine, inference "
        "model, and XAI visualizations are unified into a web-based clinical decision-support prototype built with FastAPI "
        "and React/Next.js, providing an end-to-end framework for reproducible medical imaging AI research."
    )

    # --- SECTION 2: INTRODUCTION ---
    add_h1("2. Introduction")
    add_p(
        "Alzheimer's disease is an irreversible, progressive neurodegenerative disorder that gradually destroys memory, "
        "cognitive capabilities, and the capacity to carry out simple daily tasks. As populations age globally, the prevalence "
        "of Alzheimer's disease and related dementias is projected to affect over 130 million individuals by 2050, imposing "
        "an unprecedented socio-economic burden on healthcare systems worldwide."
    )
    add_p(
        "In clinical practice, structural Magnetic Resonance Imaging (sMRI) is the non-invasive imaging modality of choice "
        "for evaluating macrostructural brain alterations associated with neurodegeneration. T1-weighted sMRI enables the "
        "visualization of anatomical hallmarks such as progressive hippocampal atrophy, entorhinal cortical thinning, and "
        "compensatory ventricular enlargement. However, manually inspecting hundreds of 2D MRI slices per patient is time-consuming, "
        "subject to inter-observer variability, and prone to missing subtle early-stage structural changes."
    )
    add_p(
        "Recent breakthroughs in deep learning and 3D Convolutional Neural Networks (3D CNNs) offer immense potential for "
        "automating the extraction of complex volumetric biomarkers directly from 3D MRI scans. Nevertheless, translating deep "
        "learning models into healthcare decision-support tools requires overcoming key challenges: preventing artificial data "
        "leakage during experimental splitting, maintaining consistent preprocessing between training and clinical serving, "
        "and mitigating the 'black-box' nature of deep neural networks through Explainable AI (XAI)."
    )

    # --- SECTION 3: BACKGROUND AND MOTIVATION ---
    add_h1("3. Background and Motivation")
    add_p(
        "The Open Access Series of Imaging Studies (OASIS-1) cross-sectional dataset is a widely recognized benchmark "
        "in neuroimaging research, containing structural T1-weighted MRI acquisitions from 413 subjects aged 18 to 96. "
        "Accompanying clinical evaluations provide detailed demographic parameters, Mini-Mental State Examination (MMSE) "
        "scores, and Clinical Dementia Rating (CDR) scales."
    )
    add_p(
        "The Clinical Dementia Rating (CDR) scale, established by Morris (1993), is a validated clinical instrument used "
        "to evaluate cognitive and functional performance across six domains: Memory, Orientation, Judgment & Problem Solving, "
        "Community Affairs, Home & Hobbies, and Personal Care. The global CDR score rates dementia severity as:"
    )
    add_bullet("Cognitively Normal (No clinically rated dementia)", bold_prefix="• CDR 0.0:")
    add_bullet("Very Mild Dementia (Equivalent to Very Mild Impairment / Early Symptomatic stage)", bold_prefix="• CDR 0.5:")
    add_bullet("Mild Dementia (Moderate cognitive impairment with functional loss)", bold_prefix="• CDR 1.0:")
    add_bullet("Moderate Dementia (Severe memory loss, highly dependent on assistance)", bold_prefix="• CDR 2.0:")
    add_p(
        "Motivated by the need for objective, reproducible stage classification, this project focuses on leveraging "
        "deep learning to map 3D sMRI volumes directly to CDR dementia severity categories, offering an automated decision-support "
        "tool for clinicians."
    )

    # --- SECTION 4: PROBLEM STATEMENT ---
    add_h1("4. Problem Statement")
    add_p("Current medical imaging AI research for dementia classification faces three critical vulnerabilities:")
    add_bullet(
        "Many published studies perform random 2D slice-wise or image-wise splitting. Because consecutive MRI slices "
        "from the same patient share near-identical anatomical features, assigning slices from one patient to both training "
        "and test sets causes severe data leakage, leading to artificially inflated accuracy that fails catastrophically on unseen clinical patients.",
        bold_prefix="1. Slice-Wise Data Leakage:"
    )
    add_bullet(
        "Inconsistencies between training-time preprocessing and deployment-time serving create train-test skew, "
        "causing deep learning models to fail when processing raw scans uploaded by clinicians.",
        bold_prefix="2. Preprocessing Discrepancies:"
    )
    add_bullet(
        "Standard deep neural networks output probability scores without highlighting which anatomical regions "
        "influenced the prediction, limiting clinician trust and clinical adoptability.",
        bold_prefix="3. Lack of Model Interpretability:"
    )

    # --- SECTION 5: RESEARCH AIM ---
    add_h1("5. Research Aim")
    add_p(
        "The primary aim of this project is to design, implement, and evaluate a reproducible, modular, explainable "
        "3D deep-learning framework and web-based decision-support prototype for four-class Alzheimer's-related dementia stage "
        "classification using structural MRI scans from the OASIS-1 cohort."
    )

    # --- SECTION 6: OBJECTIVES ---
    add_h1("6. Project Objectives")
    add_bullet("Develop a robust, reusable 13-stage modular preprocessing engine for 3D structural MRI scans.", bold_prefix="1.")
    add_bullet("Construct a strictly patient-wise partitioned supervised dataset from the OASIS-1 cohort to prevent data leakage.", bold_prefix="2.")
    add_bullet("Implement a 3D ResNet-18 deep neural network tailored for 3D volumetric MRI stage classification.", bold_prefix="3.")
    add_bullet("Output a multi-class probability distribution across four CDR categories (CDR 0, 0.5, 1, and 2).", bold_prefix="4.")
    add_bullet("Derive an aggregated overall dementia-related classification probability from non-zero CDR classes.", bold_prefix="5.")
    add_bullet("Rigorously evaluate model performance using multi-class medical machine learning metrics (ROC-AUC, F1-Score, Confusion Matrix).", bold_prefix="6.")
    add_bullet("Incorporate Grad-CAM / Grad-CAM++ Explainable AI for spatial feature attribution visualization.", bold_prefix="7.")
    add_bullet("Integrate the preprocessing pipeline, model inference, and XAI heatmaps into a FastAPI + React web prototype.", bold_prefix="8.")
    add_bullet("Preserve a permanent, reproducible research data pipeline for future comparative AI studies (2D CNNs, Vision Transformers, SSL).", bold_prefix="9.")

    # --- SECTION 7: DATASET DESCRIPTION ---
    add_h1("7. Dataset Description")
    add_p(
        "The project utilizes the Open Access Series of Imaging Studies (OASIS-1) Cross-Sectional MRI Dataset. "
        "The dataset consists of 12 disc directories containing 413 physical patient folders discovered on disk. "
        "Subject directory numbering spans up to OAS1_0457_MR1 due to 44 non-consecutive ID gaps in the original release sequence."
    )
    add_p("The dataset is partitioned into two distinct cohorts based on clinical label availability:")
    add_bullet(
        "Contains 234 subjects aged 33 to 96 years (mean age: 72.4 years) with complete CDR clinical evaluations. "
        "This cohort forms the ground-truth benchmark for supervised model training and evaluation.",
        bold_prefix="• Labelled Cohort (READY):"
    )
    add_bullet(
        "Contains 179 subjects aged 18 to 58 years (mean age: 27.3 years). These represent young, healthy baseline "
        "controls for whom CDR evaluations were not clinically administered. They are preserved in unlabelled_manifest.csv for self-supervised learning.",
        bold_prefix="• Unlabelled Cohort (MISSING_LABEL):"
    )
    add_p(
        "For each patient, the primary input volume is extracted from PROCESSED/MPRAGE/SUBJ_111/ containing gain-corrected, "
        "N4 bias-field corrected, motion-averaged 3D Analyze 7.5 MRI volumes (.hdr/.img). RAW scans, FSL segmentation outputs, "
        "and preview GIF thumbnails are intentionally excluded from primary supervised classification."
    )

    # --- SECTION 8: PROPOSED METHODOLOGY ---
    add_h1("8. Proposed Methodology")
    add_p(
        "The project architecture follows Clean Architecture (Domain-Driven Design) principles, enforcing strict separation "
        "of concerns across modular packages: dataset management, preprocessing core, model registry, training engine, "
        "inference serving, backend REST API, and frontend web UI."
    )
    add_p(
        "The core preprocessing engine is built as a shared Python library, guaranteeing that an MRI scan uploaded to "
        "the web application undergoes the exact same mathematical operations (orientation reorientation, intensity Z-score "
        "normalization, bounding-box cropping, and volumetric resizing) as training samples."
    )

    # --- SECTION 9: MRI PREPROCESSING PIPELINE ---
    add_h1("9. MRI Preprocessing Pipeline")
    add_p("The preprocessing pipeline comprises 13 sequential, deterministic operations:")
    add_bullet("Discovers all patient directories across raw disc folders.", bold_prefix="1. Dataset Scanner:")
    add_bullet("Parses OAS1_XXXX_MR1.txt files for age, gender, MMSE, and CDR labels.", bold_prefix="2. Metadata Extraction:")
    add_bullet("Loads Analyze 7.5 / NIfTI volumes into float32 NumPy arrays via NiBabel.", bold_prefix="3. MRI Loader:")
    add_bullet("Asserts 3D dimensions, non-zero foreground ratio (>5%), and checks for NaN/Inf values.", bold_prefix="4. Volume Validator:")
    add_bullet("Reorients volume affine matrices to canonical RAS (Right-Anterior-Superior) orientation.", bold_prefix="5. Spatial Orientation:")
    add_bullet("Clips 0.5–99.5% intensity outliers and applies Z-score normalization (mean=0, std=1) on non-zero brain voxels.", bold_prefix="6. Intensity Normalization:")
    add_bullet("Detects non-zero brain tissue boundaries and crops redundant background with padding.", bold_prefix="7. Brain Bounding Box Cropping:")
    add_bullet("Resizes cropped 3D volumes to standardized 128 × 128 × 128 shape using cubic spline interpolation.", bold_prefix="8. Volumetric 3D Resizing:")
    add_bullet("Extracts 2D planar slices across Axial, Coronal, and Sagittal anatomical planes.", bold_prefix="9. 2D Slice Extraction:")
    add_bullet("Calculates Shannon entropy (H ≥ 1.5) and tissue ratio (≥15%) to discard blank background slices.", bold_prefix="10. Quality Filtering:")
    add_bullet("Resizes 2D slices to 224 × 224 and serializes PyTorch tensors (tensor.pt, slices/*.npy).", bold_prefix="11. Tensor Serialization:")
    add_bullet("Generates metadata.json, preprocessing_log.json, and 3-plane diagnostic_preview.png images.", bold_prefix="12. Artifact Generation:")
    add_bullet("Executes stratified patient-wise splitting into Train, Validation, and Test manifests.", bold_prefix="13. Dataset Splitting:")

    # --- SECTION 10: TARGET VARIABLE & CLASSIFICATION STRATEGY ---
    add_h1("10. Target Variable and Classification Strategy")
    add_p(
        "The ground-truth supervised target is the Clinical Dementia Rating (CDR). The project establishes a four-class "
        "classification problem:"
    )
    add_bullet("Cognitively Normal (Index 0)", bold_prefix="• Class 0 (CDR 0.0):")
    add_bullet("Very Mild Dementia (Index 1)", bold_prefix="• Class 1 (CDR 0.5):")
    add_bullet("Mild Dementia (Index 2)", bold_prefix="• Class 2 (CDR 1.0):")
    add_bullet("Moderate Dementia (Index 3)", bold_prefix="• Class 3 (CDR 2.0):")
    add_p(
        "IMPORTANT CLINICAL INTERPRETATION: CDR is a clinical dementia-severity grading scale, not a direct molecular "
        "or histopathological diagnosis of Alzheimer's pathology (such as amyloid-beta or tau biomarker confirmation). "
        "Therefore, the task is strictly defined as Alzheimer's-related dementia stage classification."
    )
    add_p(
        "The 3D deep-learning model outputs a 4-class softmax probability vector P = [p0, p0.5, p1, p2]. The predicted stage "
        "corresponds to argmax(P). An overall dementia-related classification probability is derived by aggregating non-zero CDR probabilities:"
    )
    add_p("P(Overall Dementia-Related Classification) = P(CDR=0.5) + P(CDR=1.0) + P(CDR=2.0)")
    add_p(
        "For example, if the model predicts P = [0.04, 0.15, 0.77, 0.04], the predicted category is Mild Dementia (CDR 1, 77% confidence), "
        "and the overall dementia-related classification probability is 15% + 77% + 4% = 96%. This metric is explicitly defined as a "
        "model-derived probability aggregation, not a clinically validated Alzheimer's risk score."
    )

    # --- SECTION 11: DEEP LEARNING MODEL ARCHITECTURE ---
    add_h1("11. Deep Learning Model Architecture")
    add_p(
        "The primary deep-learning architecture selected for baseline experiments is 3D ResNet-18. 3D ResNet-18 replaces "
        "standard 2D convolutions with 3D spatial convolutions, allowing the network to process the full 128 × 128 × 128 volumetric tensor "
        "and preserve spatial anatomical relationships across all three spatial dimensions."
    )
    add_p("Key architectural advantages of 3D ResNet-18 include:")
    add_bullet("Captures inter-slice anatomical continuity and 3D structural atrophy patterns across cortical and subcortical regions.", bold_prefix="• 3D Spatial Context:")
    add_bullet("Residual skip connections prevent vanishing gradient degradation during backpropagation.", bold_prefix="• Residual Learning:")
    add_bullet("With ~14.3 million parameters, 3D ResNet-18 is computationally efficient and resistant to overfitting on medical cohorts.", bold_prefix="• Parameter Efficiency:")
    add_bullet("Fully compatible with PyTorch and CUDA acceleration on Google Colab GPU environments.", bold_prefix="• GPU Compatibility:")
    add_p(
        "Future comparative experiments will evaluate 3D DenseNet, 3D ConvNeXt, Vision Transformers (Swin3D), and 2D CNN backbones (EfficientNet-B0, DenseNet-121)."
    )

    # --- SECTION 12: EXPLAINABLE AI (XAI) ---
    add_h1("12. Explainable AI (XAI)")
    add_p(
        "To address the interpretability bottleneck in medical AI, the system incorporates Gradient-Weighted Class Activation "
        "Mapping (Grad-CAM / Grad-CAM++). Grad-CAM computes the gradients of the target CDR class score with respect to the feature "
        "maps of the final 3D convolutional layer, producing a spatial activation map that highlights regional brain features "
        "driving model predictions."
    )
    add_p(
        "CLINICAL INTERPRETATION BOUNDARY: The visual heatmaps illustrate model attention and feature contribution, "
        "not medical causality or definitive tissue pathology."
    )

    # --- SECTION 13: PROPOSED WEB APPLICATION ---
    add_h1("13. Proposed Web Application Architecture")
    add_p(
        "The system culminates in a web-based decision-support prototype. The application workflow consists of:"
    )
    add_bullet("Radiologist uploads structural MRI scan (.nii, .nii.gz, or .hdr/.img).", bold_prefix="1. File Upload:")
    add_bullet("FastAPI backend validates file structure, format, and dimensions.", bold_prefix="2. Backend Validation:")
    add_bullet("Invokes InferencePreprocessor to reorient, normalize, crop, and resize the scan into a 128×128×128 tensor.", bold_prefix="3. Shared Preprocessing:")
    add_bullet("Loaded PyTorch model computes 4-class CDR probability distribution.", bold_prefix="4. Model Inference:")
    add_bullet("Grad-CAM engine generates 3-plane spatial heatmap overlays.", bold_prefix="5. XAI Heatmap Generation:")
    add_bullet("FastAPI returns JSON prediction payload and rendered heatmap images.", bold_prefix="6. API Response:")
    add_bullet("React/Next.js UI renders an interactive patient dashboard and downloadable PDF report.", bold_prefix="7. Clinical Dashboard:")

    # --- SECTION 14: EVALUATION METHODOLOGY ---
    add_h1("14. Evaluation Methodology")
    add_p(
        "Model performance will be evaluated on the held-out test split (24 patients) using quantitative medical ML metrics:"
    )
    add_bullet("Overall proportion of correctly classified dementia stages.", bold_prefix="• Multi-class Accuracy:")
    add_bullet("Macro-averaged and weighted Precision, Sensitivity (Recall), Specificity, and F1-Score.", bold_prefix="• Precision & Recall:")
    add_bullet("Receiver Operating Characteristic curves and Area Under Curve (ROC-AUC) for multi-class classification.", bold_prefix="• ROC-AUC:")
    add_bullet("4 × 4 confusion matrix illustrating inter-class misclassification patterns between adjacent CDR stages.", bold_prefix="• Confusion Matrix:")

    # --- SECTION 15: EXPECTED OUTCOMES ---
    add_h1("15. Expected Outcomes")
    add_bullet("A fully functional, reusable 13-stage MRI preprocessing engine.", bold_prefix="1.")
    add_bullet("A leak-free, patient-wise partitioned dataset benchmark (187 Train / 23 Val / 24 Test).", bold_prefix="2.")
    add_bullet("A trained 3D ResNet-18 baseline model producing CDR stage predictions and probability distributions.", bold_prefix="3.")
    add_bullet("Grad-CAM explainability heatmaps visualizing anatomical model attention.", bold_prefix="4.")
    add_bullet("An integrated FastAPI + React clinical decision-support web application prototype.", bold_prefix="5.")

    # --- SECTION 16: RESEARCH SIGNIFICANCE ---
    add_h1("16. Research Significance")
    add_p(
        "This project contributes a rigorous, reproducible framework for neuroimaging research. By enforcing strict "
        "patient-wise splitting, it establishes a leak-free benchmark for OASIS-1 dementia stage classification. "
        "The shared preprocessing design eliminates train-test skew, while Explainable AI heatmaps enhance clinical "
        "transparency, bridging the gap between deep learning research and practical decision support."
    )

    # --- SECTION 17: LIMITATIONS ---
    add_h1("17. Limitations")
    add_bullet("Supervised training is constrained to the 234 labelled OASIS-1 subjects.", bold_prefix="1. Sample Size:")
    add_bullet("CDR is a clinical functional scale rather than a direct biomarker of Alzheimer's pathology.", bold_prefix="2. Target Nature:")
    add_bullet("OASIS-1 is a single-center cross-sectional cohort and requires multi-center external validation.", bold_prefix="3. Single-Center Cohort:")
    add_bullet("Model predictions are decision-support outputs and require expert radiological confirmation.", bold_prefix="4. Non-Diagnostic Prototype:")

    # --- SECTION 18: FUTURE SCOPE ---
    add_h1("18. Future Scope")
    add_bullet("Incorporating multi-center cohorts such as ADNI, AIBL, and OASIS-3.", bold_prefix="1. Multi-Center Expansion:")
    add_bullet("Integrating 3D MRI volumes with tabular clinical covariates (Age, MMSE, ApoE4 gene status).", bold_prefix="2. Multimodal Fusion:")
    add_bullet("Leveraging the 179 unlabelled subjects for SimCLR / Masked Autoencoder self-supervised pre-training.", bold_prefix="3. Self-Supervised Learning:")
    add_bullet("Evaluating 3D Swin Transformers and MONAI volumetric architectures.", bold_prefix="4. 3D Vision Transformers:")
    add_bullet("Integrating DICOM server communications (PACS / Orthanc) for clinical deployment.", bold_prefix="5. Clinical PACS Integration:")

    # --- SECTION 19: PROPOSED TECHNOLOGY STACK ---
    add_h1("19. Proposed Technology Stack")
    add_bullet("Core Language & Data Processing: Python 3.11, NumPy, Pandas, SciPy, scikit-image", bold_prefix="•")
    add_bullet("Neuroimaging Libraries: NiBabel, SimpleITK, MONAI framework", bold_prefix="•")
    add_bullet("Deep Learning Framework: PyTorch 2.13, Torchvision, PyTorch Lightning", bold_prefix="•")
    add_bullet("GPU Computing & Training: NVIDIA CUDA 12.0, Google Colab Pro GPU", bold_prefix="•")
    add_bullet("Explainable AI & Visualization: Grad-CAM / Grad-CAM++, Matplotlib, OpenCV", bold_prefix="•")
    add_bullet("Backend Service API: FastAPI, Uvicorn, Pydantic, ReportLab (PDF Generation)", bold_prefix="•")
    add_bullet("Frontend Web Interface: React.js / Next.js, TailwindCSS, NiftiMRI 3D Viewer", bold_prefix="•")
    add_bullet("Testing & Quality Assurance: PyTest, Flake8 code linting", bold_prefix="•")

    # --- SECTION 20: HIGH-LEVEL SYSTEM ARCHITECTURE ---
    add_h1("20. High-Level System Architecture")
    add_p("The end-to-end data flow and architectural interactions are summarized below:")

    # Table of System Architecture
    table = Table()
    for _ in range(3):
        table.addElement(TableColumn())

    # Header Row
    tr_h = TableRow()
    for title in ["System Module", "Core Components & Artifacts", "Responsibility & Data Flow"]:
        tc = TableCell()
        tc.addElement(P(stylename=th_style, text=title))
        tr_h.addElement(tc)
    table.addElement(tr_h)

    rows_data = [
        ("Dataset Scanner & Splitter", "scanner.py, split_generator.py, manifests/", "Discovers 413 patients, parses metadata, generates patient-wise 80/10/10 splits."),
        ("Preprocessing Core", "loader.py, normalizer.py, cropper.py, resizer.py", "Executes 13-stage pipeline; generates 128x128x128 3D tensors and NIfTI volumes."),
        ("PyTorch Data Pipeline", "torch_dataset.py, PyTorch DataLoader", "Feeds model-agnostic 3D [1,128,128,128] tensors to GPU training loop."),
        ("Deep Learning Model", "3D ResNet-18 (models/cnn_3d/)", "Computes 4-class CDR probability distribution across CDR 0, 0.5, 1, and 2."),
        ("Explainable AI (XAI)", "Grad-CAM / Grad-CAM++ engine", "Generates 3-plane spatial heatmap overlays highlighting model feature attention."),
        ("Inference Engine", "inference_preprocessor.py, engine.py", "Executes single-volume preprocessing & prediction for uploaded MRI scans."),
        ("Backend REST API", "FastAPI (backend/app/main.py)", "Exposes /predict, /explain, and /report endpoints for clinical web UI."),
        ("Frontend Web App", "React / Next.js Dashboard", "Provides MRI drag-and-drop upload, 3D viewer, prediction scores, and PDF report.")
    ]

    for m, c, r in rows_data:
        tr = TableRow()
        for text in [m, c, r]:
            tc = TableCell()
            tc.addElement(P(stylename=td_style, text=text))
            tr.addElement(tc)
        table.addElement(tr)

    doc.text.addElement(table)

    # --- SECTION 21: PROJECT TIMELINE / WORK PLAN ---
    add_h1("21. Project Timeline and Work Plan")
    add_bullet("Phase 1 (Months 1–2): Dataset Analysis, Preprocessing Engine, Patient-Wise Splitting, PyTest Verification. [COMPLETED]", bold_prefix="•")
    add_bullet("Phase 2 (Month 3): 3D ResNet-18 Baseline Implementation & GPU Training on Google Colab.", bold_prefix="•")
    add_bullet("Phase 3 (Month 4): Evaluation Metrics Computation, Confusion Matrix Analysis, Hyperparameter Tuning.", bold_prefix="•")
    add_bullet("Phase 4 (Month 5): Grad-CAM Explainable AI Integration & 3-Plane Heatmap Generation.", bold_prefix="•")
    add_bullet("Phase 5 (Month 6): FastAPI Backend REST Server & React/Next.js Web UI Development.", bold_prefix="•")
    add_bullet("Phase 6 (Month 7): System Testing, Final Synopsis Documentation, and Research Paper Drafting.", bold_prefix="•")

    # --- SECTION 22: CONCLUSION ---
    add_h1("22. Conclusion")
    add_p(
        "This project presents a complete, reproducible research framework for Alzheimer's-related dementia stage classification "
        "using structural MRI. By instituting a 13-stage modular preprocessing engine, strict patient-wise splitting, 3D ResNet-18 "
        "deep learning, Grad-CAM explainability, and a web decision-support prototype, the system addresses critical gaps in data "
        "leakage prevention, preprocessing consistency, and model interpretability. The outcome serves as a robust benchmark "
        "for ongoing research in computer-aided neuroimaging decision support."
    )

    # --- SECTION 23: REFERENCES ---
    add_h1("23. References")
    refs = [
        "Marcus, D. S., et al. (2007). 'Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults.' Journal of Cognitive Neuroscience, 19(9), 1498–1507.",
        "Morris, J. C. (1993). 'The Clinical Dementia Rating (CDR): current version and scoring rules.' Neurology, 43(11), 2412–2414.",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). 'Deep residual learning for image recognition.' In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770–778).",
        "Selvaraju, R. R., et al. (2017). 'Grad-CAM: Visual explanations from deep networks via gradient-based localization.' In Proceedings of the IEEE international conference on computer vision (pp. 618–626).",
        "Jack, C. R., et al. (2018). 'NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease.' Alzheimer's & Dementia, 14(4), 535–562.",
        "Tustison, N. J., et al. (2010). 'N4ITK: improved N4 bias field correction.' IEEE Transactions on Medical Imaging, 29(6), 1310–1320.",
        "Ronneberger, O., Fischer, P., & Brox, T. (2015). 'U-net: Convolutional networks for biomedical image segmentation.' In MICCAI (pp. 234–241). Springer.",
        "Dosovitskiy, A., et al. (2020). 'An image is worth 16x16 words: Transformers for image recognition at scale.' arXiv preprint arXiv:2010.11929.",
        "Liu, Z., et al. (2021). 'Swin transformer: Hierarchical vision transformer using shifted windows.' In IEEE/CVF ICCV (pp. 10012–10022).",
        "Cardoso, M. J., et al. (2022). 'MONAI: An open-source framework for deep learning in healthcare.' arXiv preprint arXiv:2211.02701."
    ]

    for ref in refs:
        add_bullet(ref, bold_prefix="[Ref]")

    # Save ODT document
    doc.save(output_filename)
    print(f"Synopsis ODT successfully generated: {output_filename}")

if __name__ == "__main__":
    create_synopsis_odt()
