"""
Robust, direct OpenDocument Text (.odt) generator script.
Creates 'Alzheimers_Dementia_Stage_Classification_Synopsis.odt'
conforming to ODF 1.2 standard.
"""

import os
import zipfile
import xml.etree.ElementTree as ET

def build_odt_synopsis(output_filename="Alzheimers_Dementia_Stage_Classification_Synopsis.odt"):
    # 1. mimetype (MUST be first entry, uncompressed)
    mimetype_content = b"application/vnd.oasis.opendocument.text"

    # 2. META-INF/manifest.xml
    manifest_xml = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
  <manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="application/vnd.oasis.opendocument.text"/>
  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

    # 3. meta.xml
    meta_xml = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/" office:version="1.2">
  <office:meta>
    <dc:title>Explainable Deep Learning for Alzheimer's-Related Dementia Stage Classification Using Structural MRI</dc:title>
    <dc:creator>Senior Academic Technical Researcher</dc:creator>
    <dc:description>Academic Project Synopsis for Computer Science &amp; Engineering Minor Project</dc:description>
  </office:meta>
</office:document-meta>"""

    # 4. styles.xml
    styles_xml = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" office:version="1.2">
  <office:styles>
    <style:default-style style:family="paragraph">
      <style:paragraph-properties fo:hyphenate="false" fo:line-height="115%" fo:space-after="0.25cm"/>
      <style:text-properties fo:font-size="11pt" style:font-name="Times New Roman" fo:color="#1e293b"/>
    </style:default-style>
  </office:styles>
  <office:automatic-styles>
    <style:page-layout style:name="StandardLayout">
      <style:page-layout-properties fo:page-width="21.0cm" fo:page-height="29.7cm" fo:margin-top="2.0cm" fo:margin-bottom="2.0cm" fo:margin-left="2.5cm" fo:margin-right="2.5cm"/>
    </style:page-layout>
  </office:automatic-styles>
  <office:master-styles>
    <style:master-page style:name="Standard" style:page-layout-name="StandardLayout"/>
  </office:master-styles>
</office:document-styles>"""

    # 5. content.xml
    content_header = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" office:version="1.2">
  <office:automatic-styles>
    <style:style style:name="DocTitle" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center" fo:space-before="1.5cm" fo:space-after="0.4cm"/>
      <style:text-properties style:font-name="Arial" fo:font-size="22pt" fo:font-weight="bold" fo:color="#1e3a8a"/>
    </style:style>
    <style:style style:name="DocSubtitle" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center" fo:space-before="0.2cm" fo:space-after="1.2cm"/>
      <style:text-properties style:font-name="Arial" fo:font-size="13pt" fo:font-style="italic" fo:color="#2563eb"/>
    </style:style>
    <style:style style:name="DocMeta" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center" fo:space-before="0.3cm" fo:space-after="0.2cm"/>
      <style:text-properties style:font-name="Arial" fo:font-size="11pt" fo:color="#475569"/>
    </style:style>
    <style:style style:name="Heading1" style:family="paragraph">
      <style:paragraph-properties fo:text-align="left" fo:space-before="0.8cm" fo:space-after="0.3cm" fo:keep-with-next="always"/>
      <style:text-properties style:font-name="Arial" fo:font-size="15pt" fo:font-weight="bold" fo:color="#1e3a8a"/>
    </style:style>
    <style:style style:name="Heading2" style:family="paragraph">
      <style:paragraph-properties fo:text-align="left" fo:space-before="0.5cm" fo:space-after="0.2cm" fo:keep-with-next="always"/>
      <style:text-properties style:font-name="Arial" fo:font-size="12.5pt" fo:font-weight="bold" fo:color="#0f766e"/>
    </style:style>
    <style:style style:name="BodyText" style:family="paragraph">
      <style:paragraph-properties fo:text-align="justify" fo:line-height="115%" fo:space-after="0.25cm"/>
      <style:text-properties style:font-name="Times New Roman" fo:font-size="11pt" fo:color="#1e293b"/>
    </style:style>
    <style:style style:name="BulletItem" style:family="paragraph">
      <style:paragraph-properties fo:text-align="justify" fo:line-height="115%" fo:space-after="0.15cm" fo:margin-left="0.6cm"/>
      <style:text-properties style:font-name="Times New Roman" fo:font-size="11pt" fo:color="#1e293b"/>
    </style:style>
    <style:style style:name="PageBreakPara" style:family="paragraph">
      <style:paragraph-properties fo:break-before="page"/>
    </style:style>
    <style:style style:name="TableHeaderCell" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center"/>
      <style:text-properties style:font-name="Arial" fo:font-size="10pt" fo:font-weight="bold" fo:color="#ffffff"/>
    </style:style>
    <style:style style:name="TableCellPara" style:family="paragraph">
      <style:paragraph-properties fo:text-align="left"/>
      <style:text-properties style:font-name="Times New Roman" fo:font-size="10pt" fo:color="#1e293b"/>
    </style:style>
    <style:style style:name="TableHeadCellProp" style:family="table-cell">
      <style:table-cell-properties fo:background-color="#1e3a8a" fo:padding="0.2cm" fo:border="0.05pt solid #cbd5e1"/>
    </style:style>
    <style:style style:name="TableCellProp" style:family="table-cell">
      <style:table-cell-properties fo:padding="0.2cm" fo:border="0.05pt solid #cbd5e1"/>
    </style:style>
    <style:style style:name="BoldSpan" style:family="text">
      <style:text-properties fo:font-weight="bold"/>
    </style:style>
  </office:automatic-styles>
  <office:body>
    <office:text>
"""

    content_footer = """
    </office:text>
  </office:body>
</office:document-content>"""

    # Build XML Body Content
    def xml_p(text, style="BodyText"):
        return f'      <text:p text:style-name="{style}">{escape_xml(text)}</text:p>\n'

    def xml_h1(text):
        return f'      <text:h text:style-name="Heading1" text:outline-level="1">{escape_xml(text)}</text:h>\n'

    def xml_h2(text):
        return f'      <text:h text:style-name="Heading2" text:outline-level="2">{escape_xml(text)}</text:h>\n'

    def xml_bullet(text, prefix=""):
        if prefix:
            prefix_xml = f'<text:span text:style-name="BoldSpan">{escape_xml(prefix)} </text:span>'
            return f'      <text:p text:style-name="BulletItem">{prefix_xml}{escape_xml(text)}</text:p>\n'
        return f'      <text:p text:style-name="BulletItem">{escape_xml(text)}</text:p>\n'

    def escape_xml(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    body_xml = []

    # Title Page
    body_xml.append(xml_p("Explainable Deep Learning for Alzheimer's-Related Dementia Stage Classification Using Structural MRI", "DocTitle"))
    body_xml.append(xml_p("An OASIS-1 Based Medical Imaging and Web-Based Decision Support System", "DocSubtitle"))
    body_xml.append(xml_p("ACADEMIC PROJECT SYNOPSIS", "DocMeta"))
    body_xml.append(xml_p("Degree: Bachelor of Technology in Computer Science & Engineering", "DocMeta"))
    body_xml.append(xml_p("Domain: Medical Imaging AI, Deep Learning, & Clinical Decision Support Systems", "DocMeta"))
    body_xml.append(xml_p("Dataset Benchmark: OASIS-1 Cross-Sectional sMRI Cohort", "DocMeta"))
    body_xml.append(xml_p("Academic Session: 2025–2026", "DocMeta"))

    # Page Break
    body_xml.append('      <text:p text:style-name="PageBreakPara"/>\n')

    # Section 1: Abstract
    body_xml.append(xml_h1("1. Abstract"))
    body_xml.append(xml_p(
        "Alzheimer's disease (AD) represents the primary cause of neurodegenerative dementia globally, characterized "
        "by progressive synaptic disruption, neuronal loss, and structural brain atrophy. Early discrimination of dementia "
        "severity stages is paramount for timely therapeutic intervention and clinical trial stratification. This project "
        "presents an explainable, 3D volumetric deep-learning framework for four-class Alzheimer's-related dementia stage "
        "classification based on structural Magnetic Resonance Imaging (sMRI) scans from the Open Access Series of Imaging "
        "Studies (OASIS-1) cross-sectional cohort. The primary objective is to classify sMRI volumes into Clinical Dementia "
        "Rating (CDR) severity stages: Cognitively Normal (CDR 0), Very Mild Dementia (CDR 0.5), Mild Dementia (CDR 1), "
        "and Moderate Dementia (CDR 2)."
    ))
    body_xml.append(xml_p(
        "The system incorporates a robust, 13-stage modular preprocessing engine that transforms raw volumetric Analyze 7.5 "
        "scans into canonical RAS-oriented, intensity-normalized (Z-score), brain-cropped 3D tensors (128 × 128 × 128) and "
        "quality-filtered 2D planar slices (224 × 224). To eliminate data leakage, a strict stratified patient-wise splitting "
        "protocol is enforced across the 234 labelled OASIS-1 subjects (80% Train / 10% Validation / 10% Test), ensuring "
        "complete subject isolation across partitions. A 3D ResNet-18 neural network serves as the primary deep-learning "
        "architecture, capturing spatial 3D anatomical relationships across the cerebral cortex and subcortical structures."
    ))
    body_xml.append(xml_p(
        "To provide interpretable insights for clinicians, Gradient-Weighted Class Activation Mapping (Grad-CAM / Grad-CAM++) "
        "is integrated to visualize salient anatomical regions driving model predictions. An overall dementia-related classification "
        "probability is derived by aggregating non-zero CDR class probabilities. Finally, the preprocessing engine, inference "
        "model, and XAI visualizations are unified into a web-based clinical decision-support prototype built with FastAPI "
        "and React/Next.js, providing an end-to-end framework for reproducible medical imaging AI research."
    ))

    # Section 2: Introduction
    body_xml.append(xml_h1("2. Introduction"))
    body_xml.append(xml_p(
        "Alzheimer's disease is an irreversible, progressive neurodegenerative disorder that gradually destroys memory, "
        "cognitive capabilities, and the capacity to carry out simple daily tasks. As populations age globally, the prevalence "
        "of Alzheimer's disease and related dementias is projected to affect over 130 million individuals by 2050, imposing "
        "an unprecedented socio-economic burden on healthcare systems worldwide."
    ))
    body_xml.append(xml_p(
        "In clinical practice, structural Magnetic Resonance Imaging (sMRI) is the non-invasive imaging modality of choice "
        "for evaluating macrostructural brain alterations associated with neurodegeneration. T1-weighted sMRI enables the "
        "visualization of anatomical hallmarks such as progressive hippocampal atrophy, entorhinal cortical thinning, and "
        "compensatory ventricular enlargement. However, manually inspecting hundreds of 2D MRI slices per patient is time-consuming, "
        "subject to inter-observer variability, and prone to missing subtle early-stage structural changes."
    ))
    body_xml.append(xml_p(
        "Recent breakthroughs in deep learning and 3D Convolutional Neural Networks (3D CNNs) offer immense potential for "
        "automating the extraction of complex volumetric biomarkers directly from 3D MRI scans. Nevertheless, translating deep "
        "learning models into healthcare decision-support tools requires overcoming key challenges: preventing artificial data "
        "leakage during experimental splitting, maintaining consistent preprocessing between training and clinical serving, "
        "and mitigating the 'black-box' nature of deep neural networks through Explainable AI (XAI)."
    ))

    # Section 3: Background & Motivation
    body_xml.append(xml_h1("3. Background and Motivation"))
    body_xml.append(xml_p(
        "The Open Access Series of Imaging Studies (OASIS-1) cross-sectional dataset is a widely recognized benchmark "
        "in neuroimaging research, containing structural T1-weighted MRI acquisitions from 413 subjects aged 18 to 96. "
        "Accompanying clinical evaluations provide detailed demographic parameters, Mini-Mental State Examination (MMSE) "
        "scores, and Clinical Dementia Rating (CDR) scales."
    ))
    body_xml.append(xml_p(
        "The Clinical Dementia Rating (CDR) scale, established by Morris (1993), is a validated clinical instrument used "
        "to evaluate cognitive and functional performance across six domains: Memory, Orientation, Judgment & Problem Solving, "
        "Community Affairs, Home & Hobbies, and Personal Care. The global CDR score rates dementia severity as:"
    ))
    body_xml.append(xml_bullet("Cognitively Normal (No clinically rated dementia)", "• CDR 0.0:"))
    body_xml.append(xml_bullet("Very Mild Dementia (Equivalent to Very Mild Impairment / Early Symptomatic stage)", "• CDR 0.5:"))
    body_xml.append(xml_bullet("Mild Dementia (Moderate cognitive impairment with functional loss)", "• CDR 1.0:"))
    body_xml.append(xml_bullet("Moderate Dementia (Severe memory loss, highly dependent on assistance)", "• CDR 2.0:"))
    body_xml.append(xml_p(
        "Motivated by the need for objective, reproducible stage classification, this project focuses on leveraging "
        "deep learning to map 3D sMRI volumes directly to CDR dementia severity categories, offering an automated decision-support "
        "tool for clinicians."
    ))

    # Section 4: Problem Statement
    body_xml.append(xml_h1("4. Problem Statement"))
    body_xml.append(xml_p("Current medical imaging AI research for dementia classification faces three critical vulnerabilities:"))
    body_xml.append(xml_bullet(
        "Many published studies perform random 2D slice-wise or image-wise splitting. Because consecutive MRI slices "
        "from the same patient share near-identical anatomical features, assigning slices from one patient to both training "
        "and test sets causes severe data leakage, leading to artificially inflated accuracy that fails catastrophically on unseen clinical patients.",
        "1. Slice-Wise Data Leakage:"
    ))
    body_xml.append(xml_bullet(
        "Inconsistencies between training-time preprocessing and deployment-time serving create train-test skew, "
        "causing deep learning models to fail when processing raw scans uploaded by clinicians.",
        "2. Preprocessing Discrepancies:"
    ))
    body_xml.append(xml_bullet(
        "Standard deep neural networks output probability scores without highlighting which anatomical regions "
        "influenced the prediction, limiting clinician trust and clinical adoptability.",
        "3. Lack of Model Interpretability:"
    ))

    # Section 5: Research Aim
    body_xml.append(xml_h1("5. Research Aim"))
    body_xml.append(xml_p(
        "The primary aim of this project is to design, implement, and evaluate a reproducible, modular, explainable "
        "3D deep-learning framework and web-based decision-support prototype for four-class Alzheimer's-related dementia stage "
        "classification using structural MRI scans from the OASIS-1 cohort."
    ))

    # Section 6: Objectives
    body_xml.append(xml_h1("6. Project Objectives"))
    body_xml.append(xml_bullet("Develop a robust, reusable 13-stage modular preprocessing engine for 3D structural MRI scans.", "1."))
    body_xml.append(xml_bullet("Construct a strictly patient-wise partitioned supervised dataset from the OASIS-1 cohort to prevent data leakage.", "2."))
    body_xml.append(xml_bullet("Implement a 3D ResNet-18 deep neural network tailored for 3D volumetric MRI stage classification.", "3."))
    body_xml.append(xml_bullet("Output a multi-class probability distribution across four CDR categories (CDR 0, 0.5, 1, and 2).", "4."))
    body_xml.append(xml_bullet("Derive an aggregated overall dementia-related classification probability from non-zero CDR classes.", "5."))
    body_xml.append(xml_bullet("Rigorously evaluate model performance using multi-class medical machine learning metrics (ROC-AUC, F1-Score, Confusion Matrix).", "6."))
    body_xml.append(xml_bullet("Incorporate Grad-CAM / Grad-CAM++ Explainable AI for spatial feature attribution visualization.", "7."))
    body_xml.append(xml_bullet("Integrate the preprocessing pipeline, model inference, and XAI heatmaps into a FastAPI + React web prototype.", "8."))
    body_xml.append(xml_bullet("Establish a permanent, reproducible research data pipeline for future comparative AI studies (2D CNNs, Vision Transformers, SSL).", "9."))

    # Section 7: Dataset Description
    body_xml.append(xml_h1("7. Dataset Description"))
    body_xml.append(xml_p(
        "The project utilizes the Open Access Series of Imaging Studies (OASIS-1) Cross-Sectional MRI Dataset. "
        "The dataset consists of 12 disc directories containing 413 physical patient folders discovered on disk. "
        "Subject directory numbering spans up to OAS1_0457_MR1 due to 44 non-consecutive ID gaps in the original release sequence."
    ))
    body_xml.append(xml_p("The dataset is partitioned into two distinct cohorts based on clinical label availability:"))
    body_xml.append(xml_bullet(
        "Contains 234 subjects aged 33 to 96 years (mean age: 72.4 years) with complete CDR clinical evaluations. "
        "This cohort forms the ground-truth benchmark for supervised model training and evaluation.",
        "• Labelled Cohort (READY):"
    ))
    body_xml.append(xml_bullet(
        "Contains 179 subjects aged 18 to 58 years (mean age: 27.3 years). These represent young, healthy baseline "
        "controls for whom CDR evaluations were not clinically administered. They are preserved in unlabelled_manifest.csv for self-supervised learning.",
        "• Unlabelled Cohort (MISSING_LABEL):"
    ))
    body_xml.append(xml_p(
        "For each patient, the primary input volume is extracted from PROCESSED/MPRAGE/SUBJ_111/ containing gain-corrected, "
        "N4 bias-field corrected, motion-averaged 3D Analyze 7.5 MRI volumes (.hdr/.img). RAW scans, FSL segmentation outputs, "
        "and preview GIF thumbnails are intentionally excluded from primary supervised classification."
    ))

    # Section 8: Proposed Methodology
    body_xml.append(xml_h1("8. Proposed Methodology"))
    body_xml.append(xml_p(
        "The project architecture follows Clean Architecture (Domain-Driven Design) principles, enforcing strict separation "
        "of concerns across modular packages: dataset management, preprocessing core, model registry, training engine, "
        "inference serving, backend REST API, and frontend web UI."
    ))
    body_xml.append(xml_p(
        "The core preprocessing engine is built as a shared Python library, guaranteeing that an MRI scan uploaded to "
        "the web application undergoes the exact same mathematical operations (orientation reorientation, intensity Z-score "
        "normalization, bounding-box cropping, and volumetric resizing) as training samples."
    ))

    # Section 9: Preprocessing Pipeline
    body_xml.append(xml_h1("9. MRI Preprocessing Pipeline"))
    body_xml.append(xml_p("The preprocessing pipeline comprises 13 sequential, deterministic operations:"))
    body_xml.append(xml_bullet("Discovers all patient directories across raw disc folders.", "1. Dataset Scanner:"))
    body_xml.append(xml_bullet("Parses OAS1_XXXX_MR1.txt files for age, gender, MMSE, and CDR labels.", "2. Metadata Extraction:"))
    body_xml.append(xml_bullet("Loads Analyze 7.5 / NIfTI volumes into float32 NumPy arrays via NiBabel.", "3. MRI Loader:"))
    body_xml.append(xml_bullet("Asserts 3D dimensions, non-zero foreground ratio (>5%), and checks for NaN/Inf values.", "4. Volume Validator:"))
    body_xml.append(xml_bullet("Reorients volume affine matrices to canonical RAS (Right-Anterior-Superior) orientation.", "5. Spatial Orientation:"))
    body_xml.append(xml_bullet("Clips 0.5–99.5% intensity outliers and applies Z-score normalization (mean=0, std=1) on non-zero brain voxels.", "6. Intensity Normalization:"))
    body_xml.append(xml_bullet("Detects non-zero brain tissue boundaries and crops redundant background with padding.", "7. Brain Bounding Box Cropping:"))
    body_xml.append(xml_bullet("Resizes cropped 3D volumes to standardized 128 × 128 × 128 shape using cubic spline interpolation.", "8. Volumetric 3D Resizing:"))
    body_xml.append(xml_bullet("Extracts 2D planar slices across Axial, Coronal, and Sagittal anatomical planes.", "9. 2D Slice Extraction:"))
    body_xml.append(xml_bullet("Calculates Shannon entropy (H ≥ 1.5) and tissue ratio (≥15%) to discard blank background slices.", "10. Quality Filtering:"))
    body_xml.append(xml_bullet("Resizes 2D slices to 224 × 224 and serializes PyTorch tensors (tensor.pt, slices/*.npy).", "11. Tensor Serialization:"))
    body_xml.append(xml_bullet("Generates metadata.json, preprocessing_log.json, and 3-plane diagnostic_preview.png images.", "12. Artifact Generation:"))
    body_xml.append(xml_bullet("Executes stratified patient-wise splitting into Train, Validation, and Test manifests.", "13. Dataset Splitting:"))

    # Section 10: Target Variable & Classification Strategy
    body_xml.append(xml_h1("10. Target Variable and Classification Strategy"))
    body_xml.append(xml_p(
        "The ground-truth supervised target is the Clinical Dementia Rating (CDR). The project establishes a four-class "
        "classification problem:"
    ))
    body_xml.append(xml_bullet("Cognitively Normal (Index 0)", "• Class 0 (CDR 0.0):"))
    body_xml.append(xml_bullet("Very Mild Dementia (Index 1)", "• Class 1 (CDR 0.5):"))
    body_xml.append(xml_bullet("Mild Dementia (Index 2)", "• Class 2 (CDR 1.0):"))
    body_xml.append(xml_bullet("Moderate Dementia (Index 3)", "• Class 3 (CDR 2.0):"))
    body_xml.append(xml_p(
        "IMPORTANT CLINICAL INTERPRETATION: CDR is a clinical dementia-severity grading scale, not a direct molecular "
        "or histopathological diagnosis of Alzheimer's pathology (such as amyloid-beta or tau biomarker confirmation). "
        "Therefore, the task is strictly defined as Alzheimer's-related dementia stage classification."
    ))
    body_xml.append(xml_p(
        "The 3D deep-learning model outputs a 4-class softmax probability vector P = [p0, p0.5, p1, p2]. The predicted stage "
        "corresponds to argmax(P). An overall dementia-related classification probability is derived by aggregating non-zero CDR probabilities:"
    ))
    body_xml.append(xml_p("P(Overall Dementia-Related Classification) = P(CDR=0.5) + P(CDR=1.0) + P(CDR=2.0)"))
    body_xml.append(xml_p(
        "For example, if the model predicts P = [0.04, 0.15, 0.77, 0.04], the predicted category is Mild Dementia (CDR 1, 77% confidence), "
        "and the overall dementia-related classification probability is 15% + 77% + 4% = 96%. This metric is explicitly defined as a "
        "model-derived probability aggregation, not a clinically validated Alzheimer's risk score."
    ))

    # Section 11: Deep Learning Model Architecture
    body_xml.append(xml_h1("11. Deep Learning Model Architecture"))
    body_xml.append(xml_p(
        "The primary deep-learning architecture selected for baseline experiments is 3D ResNet-18. 3D ResNet-18 replaces "
        "standard 2D convolutions with 3D spatial convolutions, allowing the network to process the full 128 × 128 × 128 volumetric tensor "
        "and preserve spatial anatomical relationships across all three spatial dimensions."
    ))
    body_xml.append(xml_p("Key architectural advantages of 3D ResNet-18 include:"))
    body_xml.append(xml_bullet("Captures inter-slice anatomical continuity and 3D structural atrophy patterns across cortical and subcortical regions.", "• 3D Spatial Context:"))
    body_xml.append(xml_bullet("Residual skip connections prevent vanishing gradient degradation during backpropagation.", "• Residual Learning:"))
    body_xml.append(xml_bullet("With ~14.3 million parameters, 3D ResNet-18 is computationally efficient and resistant to overfitting on medical cohorts.", "• Parameter Efficiency:"))
    body_xml.append(xml_bullet("Fully compatible with PyTorch and CUDA acceleration on Google Colab GPU environments.", "• GPU Compatibility:"))
    body_xml.append(xml_p(
        "Future comparative experiments will evaluate 3D DenseNet, 3D ConvNeXt, Vision Transformers (Swin3D), and 2D CNN backbones (EfficientNet-B0, DenseNet-121)."
    ))

    # Section 12: Explainable AI (XAI)
    body_xml.append(xml_h1("12. Explainable AI (XAI)"))
    body_xml.append(xml_p(
        "To address the interpretability bottleneck in medical AI, the system incorporates Gradient-Weighted Class Activation "
        "Mapping (Grad-CAM / Grad-CAM++). Grad-CAM computes the gradients of the target CDR class score with respect to the feature "
        "maps of the final 3D convolutional layer, producing a spatial activation map that highlights regional brain features "
        "driving model predictions."
    ))
    body_xml.append(xml_p(
        "CLINICAL INTERPRETATION BOUNDARY: The visual heatmaps illustrate model attention and feature contribution, "
        "not medical causality or definitive tissue pathology."
    ))

    # Section 13: Proposed Web Application
    body_xml.append(xml_h1("13. Proposed Web Application Architecture"))
    body_xml.append(xml_p("The system culminates in a web-based decision-support prototype. The application workflow consists of:"))
    body_xml.append(xml_bullet("Radiologist uploads structural MRI scan (.nii, .nii.gz, or .hdr/.img).", "1. File Upload:"))
    body_xml.append(xml_bullet("FastAPI backend validates file structure, format, and dimensions.", "2. Backend Validation:"))
    body_xml.append(xml_bullet("Invokes InferencePreprocessor to reorient, normalize, crop, and resize the scan into a 128×128×128 tensor.", "3. Shared Preprocessing:"))
    body_xml.append(xml_bullet("Loaded PyTorch model computes 4-class CDR probability distribution.", "4. Model Inference:"))
    body_xml.append(xml_bullet("Grad-CAM engine generates 3-plane spatial heatmap overlays.", "5. XAI Heatmap Generation:"))
    body_xml.append(xml_bullet("FastAPI returns JSON prediction payload and rendered heatmap images.", "6. API Response:"))
    body_xml.append(xml_bullet("React/Next.js UI renders an interactive patient dashboard and downloadable PDF report.", "7. Clinical Dashboard:"))

    # Section 14: Evaluation Methodology
    body_xml.append(xml_h1("14. Evaluation Methodology"))
    body_xml.append(xml_p("Model performance will be evaluated on the held-out test split (24 patients) using quantitative medical ML metrics:"))
    body_xml.append(xml_bullet("Overall proportion of correctly classified dementia stages.", "• Multi-class Accuracy:"))
    body_xml.append(xml_bullet("Macro-averaged and weighted Precision, Sensitivity (Recall), Specificity, and F1-Score.", "• Precision & Recall:"))
    body_xml.append(xml_bullet("Receiver Operating Characteristic curves and Area Under Curve (ROC-AUC) for multi-class classification.", "• ROC-AUC:"))
    body_xml.append(xml_bullet("4 × 4 confusion matrix illustrating inter-class misclassification patterns between adjacent CDR stages.", "• Confusion Matrix:"))

    # Section 15: Expected Outcomes
    body_xml.append(xml_h1("15. Expected Outcomes"))
    body_xml.append(xml_bullet("A fully functional, reusable 13-stage MRI preprocessing engine.", "1."))
    body_xml.append(xml_bullet("A leak-free, patient-wise partitioned dataset benchmark (187 Train / 23 Val / 24 Test).", "2."))
    body_xml.append(xml_bullet("A trained 3D ResNet-18 baseline model producing CDR stage predictions and probability distributions.", "3."))
    body_xml.append(xml_bullet("Grad-CAM explainability heatmaps visualizing anatomical model attention.", "4."))
    body_xml.append(xml_bullet("An integrated FastAPI + React clinical decision-support web application prototype.", "5."))

    # Section 16: Research Significance
    body_xml.append(xml_h1("16. Research Significance"))
    body_xml.append(xml_p(
        "This project contributes a rigorous, reproducible framework for neuroimaging research. By enforcing strict "
        "patient-wise splitting, it establishes a leak-free benchmark for OASIS-1 dementia stage classification. "
        "The shared preprocessing design eliminates train-test skew, while Explainable AI heatmaps enhance clinical "
        "transparency, bridging the gap between deep learning research and practical decision support."
    ))

    # Section 17: Limitations
    body_xml.append(xml_h1("17. Limitations"))
    body_xml.append(xml_bullet("Supervised training is constrained to the 234 labelled OASIS-1 subjects.", "1. Sample Size:"))
    body_xml.append(xml_bullet("CDR is a clinical functional scale rather than a direct biomarker of Alzheimer's pathology.", "2. Target Nature:"))
    body_xml.append(xml_bullet("OASIS-1 is a single-center cross-sectional cohort and requires multi-center external validation.", "3. Single-Center Cohort:"))
    body_xml.append(xml_bullet("Model predictions are decision-support outputs and require expert radiological confirmation.", "4. Non-Diagnostic Prototype:"))

    # Section 18: Future Scope
    body_xml.append(xml_h1("18. Future Scope"))
    body_xml.append(xml_bullet("Incorporating multi-center cohorts such as ADNI, AIBL, and OASIS-3.", "1. Multi-Center Expansion:"))
    body_xml.append(xml_bullet("Integrating 3D MRI volumes with tabular clinical covariates (Age, MMSE, ApoE4 gene status).", "2. Multimodal Fusion:"))
    body_xml.append(xml_bullet("Leveraging the 179 unlabelled subjects for SimCLR / Masked Autoencoder self-supervised pre-training.", "3. Self-Supervised Learning:"))
    body_xml.append(xml_bullet("Evaluating 3D Swin Transformers and MONAI volumetric architectures.", "4. 3D Vision Transformers:"))
    body_xml.append(xml_bullet("Integrating DICOM server communications (PACS / Orthanc) for clinical deployment.", "5. Clinical PACS Integration:"))

    # Section 19: Proposed Technology Stack
    body_xml.append(xml_h1("19. Proposed Technology Stack"))
    body_xml.append(xml_bullet("Core Language & Data Processing: Python 3.11, NumPy, Pandas, SciPy, scikit-image", "•"))
    body_xml.append(xml_bullet("Neuroimaging Libraries: NiBabel, SimpleITK, MONAI framework", "•"))
    body_xml.append(xml_bullet("Deep Learning Framework: PyTorch 2.13, Torchvision, PyTorch Lightning", "•"))
    body_xml.append(xml_bullet("GPU Computing & Training: NVIDIA CUDA 12.0, Google Colab Pro GPU", "•"))
    body_xml.append(xml_bullet("Explainable AI & Visualization: Grad-CAM / Grad-CAM++, Matplotlib, OpenCV", "•"))
    body_xml.append(xml_bullet("Backend Service API: FastAPI, Uvicorn, Pydantic, ReportLab (PDF Generation)", "•"))
    body_xml.append(xml_bullet("Frontend Web Interface: React.js / Next.js, TailwindCSS, NiftiMRI 3D Viewer", "•"))
    body_xml.append(xml_bullet("Testing & Quality Assurance: PyTest, Flake8 code linting", "•"))

    # Section 20: System Architecture Table
    body_xml.append(xml_h1("20. High-Level System Architecture"))
    body_xml.append(xml_p("The end-to-end data flow and architectural interactions are summarized below:"))

    # ODF Table Construction
    table_xml = []
    table_xml.append('      <table:table table:name="ArchitectureTable">\n')
    table_xml.append('        <table:table-column table:number-columns-repeated="3"/>\n')

    # Header Row
    table_xml.append('        <table:table-row>\n')
    for h in ["System Module", "Core Components & Artifacts", "Responsibility & Data Flow"]:
        table_xml.append(f'          <table:table-cell table:style-name="TableHeadCellProp"><text:p text:style-name="TableHeaderCell">{escape_xml(h)}</text:p></table:table-cell>\n')
    table_xml.append('        </table:table-row>\n')

    rows = [
        ("Dataset Scanner & Splitter", "scanner.py, split_generator.py, manifests/", "Discovers 413 patients, parses metadata, generates patient-wise 80/10/10 splits."),
        ("Preprocessing Core", "loader.py, normalizer.py, cropper.py, resizer.py", "Executes 13-stage pipeline; generates 128x128x128 3D tensors and NIfTI volumes."),
        ("PyTorch Data Pipeline", "torch_dataset.py, PyTorch DataLoader", "Feeds model-agnostic 3D [1,128,128,128] tensors to GPU training loop."),
        ("Deep Learning Model", "3D ResNet-18 (models/cnn_3d/)", "Computes 4-class CDR probability distribution across CDR 0, 0.5, 1, and 2."),
        ("Explainable AI (XAI)", "Grad-CAM / Grad-CAM++ engine", "Generates 3-plane spatial heatmap overlays highlighting model feature attention."),
        ("Inference Engine", "inference_preprocessor.py, engine.py", "Executes single-volume preprocessing & prediction for uploaded MRI scans."),
        ("Backend REST API", "FastAPI (backend/app/main.py)", "Exposes /predict, /explain, and /report endpoints for clinical web UI."),
        ("Frontend Web App", "React / Next.js Dashboard", "Provides MRI drag-and-drop upload, 3D viewer, prediction scores, and PDF report.")
    ]

    for m, c, r in rows:
        table_xml.append('        <table:table-row>\n')
        table_xml.append(f'          <table:table-cell table:style-name="TableCellProp"><text:p text:style-name="TableCellPara">{escape_xml(m)}</text:p></table:table-cell>\n')
        table_xml.append(f'          <table:table-cell table:style-name="TableCellProp"><text:p text:style-name="TableCellPara">{escape_xml(c)}</text:p></table:table-cell>\n')
        table_xml.append(f'          <table:table-cell table:style-name="TableCellProp"><text:p text:style-name="TableCellPara">{escape_xml(r)}</text:p></table:table-cell>\n')
        table_xml.append('        </table:table-row>\n')

    table_xml.append('      </table>:table>\n')
    body_xml.append("".join(table_xml))

    # Section 21: Timeline
    body_xml.append(xml_h1("21. Project Timeline and Work Plan"))
    body_xml.append(xml_bullet("Phase 1 (Months 1–2): Dataset Analysis, Preprocessing Engine, Patient-Wise Splitting, PyTest Verification. [COMPLETED]", "•"))
    body_xml.append(xml_bullet("Phase 2 (Month 3): 3D ResNet-18 Baseline Implementation & GPU Training on Google Colab.", "•"))
    body_xml.append(xml_bullet("Phase 3 (Month 4): Evaluation Metrics Computation, Confusion Matrix Analysis, Hyperparameter Tuning.", "•"))
    body_xml.append(xml_bullet("Phase 4 (Month 5): Grad-CAM Explainable AI Integration & 3-Plane Heatmap Generation.", "•"))
    body_xml.append(xml_bullet("Phase 5 (Month 6): FastAPI Backend REST Server & React/Next.js Web UI Development.", "•"))
    body_xml.append(xml_bullet("Phase 6 (Month 7): System Testing, Final Synopsis Documentation, and Research Paper Drafting.", "•"))

    # Section 22: Conclusion
    body_xml.append(xml_h1("22. Conclusion"))
    body_xml.append(xml_p(
        "This project presents a complete, reproducible research framework for Alzheimer's-related dementia stage classification "
        "using structural MRI. By instituting a 13-stage modular preprocessing engine, strict patient-wise splitting, 3D ResNet-18 "
        "deep learning, Grad-CAM explainability, and a web decision-support prototype, the system addresses critical gaps in data "
        "leakage prevention, preprocessing consistency, and model interpretability. The outcome serves as a robust benchmark "
        "for ongoing research in computer-aided neuroimaging decision support."
    ))

    # Section 23: References
    body_xml.append(xml_h1("23. References"))
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
    for r in refs:
        body_xml.append(xml_bullet(r, "[Ref]"))

    full_content_xml = content_header + "".join(body_xml) + content_footer

    # 6. Create ZIP Archive for ODT
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write uncompressed mimetype first
        zf.writestr("mimetype", mimetype_content, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.xml", manifest_xml.encode("utf-8"))
        zf.writestr("meta.xml", meta_xml.encode("utf-8"))
        zf.writestr("styles.xml", styles_xml.encode("utf-8"))
        zf.writestr("content.xml", full_content_xml.encode("utf-8"))

    print(f"Successfully generated clean ODT document: {output_filename}")

if __name__ == "__main__":
    build_odt_synopsis()
