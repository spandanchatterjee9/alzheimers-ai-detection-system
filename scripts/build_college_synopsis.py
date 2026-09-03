"""
Master Synopsis Generator aligned strictly with the official college-provided structure:
1. Title Page
2. 1. Introduction
3. 2. Literature Review (BRIEF)
4. 3. Problem Statement
5. 4. Objectives of the Project
6. 5. Methodology (Reflecting 2D EfficientNet-B0)
7. 6. References

Generates:
- Alzheimers_Dementia_Stage_Classification_Synopsis.docx
- Alzheimers_Dementia_Stage_Classification_Synopsis.odt
- Alzheimers_Dementia_Stage_Classification_Synopsis.pdf
"""

import os
import sys
import zipfile
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets cell background color in docx."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def generate_college_docx_synopsis(output_filename="Alzheimers_Dementia_Stage_Classification_Synopsis.docx"):
    doc = Document()

    # Page Margins (1 inch / 2.54cm all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A) # Navy Blue
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x76, 0x6E) # Teal
        return p

    def add_p(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        return p

    def add_bullet(text, prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        if prefix:
            run_p = p.add_run(prefix + " ")
            run_p.bold = True
            run_p.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.add_run(text)
        return p

    # =========================================================
    # TITLE PAGE (College Format)
    # =========================================================
    p_main_title = doc.add_paragraph()
    p_main_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_main_title.paragraph_format.space_before = Pt(24)
    p_main_title.paragraph_format.space_after = Pt(8)
    run_t = p_main_title.add_run("Explainable Deep Learning for Alzheimer's-Related Dementia Stage Classification Using Structural MRI")
    run_t.font.name = 'Arial'
    run_t.font.size = Pt(20)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(2)
    p_sub.paragraph_format.space_after = Pt(24)
    run_sub = p_sub.add_run("An OASIS-1 Based Medical Imaging and Web-Based Decision Support System")
    run_sub.font.name = 'Arial'
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

    p_meta_heading = doc.add_paragraph()
    p_meta_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta_heading.paragraph_format.space_after = Pt(16)
    r_meta_h = p_meta_heading.add_run("MINOR PROJECT SYNOPSIS SUBMISSION")
    r_meta_h.font.name = 'Arial'
    r_meta_h.font.size = Pt(13)
    r_meta_h.font.bold = True
    r_meta_h.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    # Team Members Block
    p_team_h = doc.add_paragraph()
    p_team_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_team_h.paragraph_format.space_after = Pt(4)
    r_th = p_team_h.add_run("Submitted By (Team Members & Registration Numbers):")
    r_th.font.name = 'Arial'
    r_th.font.size = Pt(11)
    r_th.font.bold = True

    members = [
        "Spandan Chatterjee (Reg. No. 230905488) – Department of Computer Science & Engineering",
        "Sambit Manna (Reg. No. 230905181) – Department of Computer Science & Engineering",
        "Guru Asrith Nakka (Reg. No. 230905182) – Department of Computer Science & Engineering",
        "Tanishque Srivastava (Reg. No. 220905123) – Department of Computer Science & Engineering"
    ]
    for m in members:
        pm = doc.add_paragraph()
        pm.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pm.paragraph_format.space_after = Pt(2)
        rm = pm.add_run(m)
        rm.font.name = 'Times New Roman'
        rm.font.size = Pt(10.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Institutional Info Block
    inst_info = [
        ("Minor Specialization:", "Artificial Intelligence & Machine Learning"),
        ("Department Offering:", "Department of Computer Science & Engineering"),
        ("Institution:", "Manipal Institute of Technology (MIT), Manipal"),
        ("Faculty Guide / Mentor:", "[Faculty Guide Name]"),
        ("Date of Submission:", "15th March 2026")
    ]
    for label, val in inst_info:
        pi = doc.add_paragraph()
        pi.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pi.paragraph_format.space_after = Pt(3)
        r_lbl = pi.add_run(f"{label} ")
        r_lbl.bold = True
        r_lbl.font.name = 'Arial'
        r_lbl.font.size = Pt(10.5)
        r_val = pi.add_run(val)
        r_val.font.name = 'Times New Roman'
        r_val.font.size = Pt(10.5)

    doc.add_page_break()

    # =========================================================
    # SECTION 1: INTRODUCTION
    # =========================================================
    add_h1("1. Introduction")
    add_p(
        "Alzheimer's disease (AD) represents the primary cause of neurodegenerative dementia globally, characterized "
        "by progressive synaptic disruption, neuronal loss, and macrostructural brain atrophy. As populations age globally, "
        "the prevalence of Alzheimer's disease and related dementias is projected to affect over 130 million individuals by 2050, "
        "imposing an unprecedented socio-economic burden on healthcare systems worldwide. Early discrimination of dementia "
        "severity stages is paramount for timely therapeutic intervention, patient management, and clinical trial stratification."
    )
    add_p(
        "In clinical practice, structural Magnetic Resonance Imaging (sMRI) is the non-invasive imaging modality of choice "
        "for evaluating macrostructural brain alterations associated with neurodegeneration. T1-weighted sMRI enables the "
        "visualization of anatomical hallmarks such as progressive hippocampal atrophy, entorhinal cortical thinning, and "
        "compensatory ventricular enlargement. However, manually inspecting hundreds of 2D MRI slices per patient is time-consuming, "
        "subject to inter-observer variability, and prone to missing subtle early-stage structural changes."
    )
    add_p(
        "To address these challenges, this project presents an explainable, 2D slice-based deep-learning framework for four-class "
        "Alzheimer's-related dementia stage classification using structural sMRI scans from the Open Access Series of Imaging Studies "
        "(OASIS-1) cross-sectional cohort. The primary goal is to classify sMRI representations into Clinical Dementia Rating (CDR) "
        "severity stages: Cognitively Normal (CDR 0), Very Mild Dementia (CDR 0.5), Mild Dementia (CDR 1), and Moderate Dementia (CDR 2)."
    )
    add_p(
        "To maximize computational efficiency, reduce GPU memory requirements, and accelerate model training while retaining "
        "critical anatomical features, the primary deep-learning pipeline utilizes 2D EfficientNet-B0 operating on standardized "
        "2D MRI slices (224 × 224) extracted across the Axial anatomical plane. Predictions generated at the slice level are "
        "aggregated to the patient level via Mean Probability Aggregation. Furthermore, Gradient-Weighted Class Activation Mapping "
        "(2D Grad-CAM / Grad-CAM++) is integrated to visualize salient anatomical features driving model predictions. Finally, "
        "the preprocessing engine, 2D inference model, and XAI visualizations are unified into a web-based clinical decision-support "
        "prototype built with FastAPI and React/Next.js."
    )

    # =========================================================
    # SECTION 2: LITERATURE REVIEW (BRIEF)
    # =========================================================
    add_h1("2. Literature Review")
    add_p(
        "Neuroimaging-based computer-aided diagnosis of dementia has evolved significantly with advances in machine learning "
        "and deep neural networks. Marcus et al. (2007) introduced the Open Access Series of Imaging Studies (OASIS-1) cross-sectional "
        "cohort, establishing a benchmark dataset of T1-weighted sMRI scans across non-demented and demented individuals aged 18 to 96. "
        "Morris (1993) formalized the Clinical Dementia Rating (CDR) scale, providing a validated clinical scoring system across six "
        "functional domains to grade dementia severity as Normal (CDR 0), Very Mild (CDR 0.5), Mild (CDR 1), and Moderate (CDR 2)."
    )
    add_p(
        "In deep learning literature, Convolutional Neural Networks (CNNs) have shown strong capability in extracting volumetric "
        "and spatial biomarkers. He et al. (2016) introduced Deep Residual Learning (ResNet), mitigating vanishing gradient degradation "
        "in deep networks. Tan & Le (2019) developed EfficientNet, demonstrating that compound scaling of network depth, width, "
        "and resolution achieves superior feature representation efficiency with significantly fewer parameters compared to standard CNNs. "
        "For model interpretability, Selvaraju et al. (2017) formulated Gradient-Weighted Class Activation Mapping (Grad-CAM), "
        "enabling visual localization of salient image regions influencing network predictions."
    )
    add_p(
        "Recent clinical frameworks (Jack et al., 2018; Cardoso et al., 2022) emphasize the necessity of combining high-efficiency "
        "deep neural networks with rigorous data partitioning and explainability tools to ensure reproducible and interpretable "
        "decision-support systems in neuroimaging research."
    )

    # =========================================================
    # SECTION 3: PROBLEM STATEMENT
    # =========================================================
    add_h1("3. Problem Statement")
    add_p("Current medical imaging AI research for dementia stage classification faces three critical vulnerabilities:")
    add_bullet(
        "Many published studies perform random 2D slice-wise or image-wise splitting. Because consecutive MRI slices "
        "from the same patient share near-identical anatomical features, assigning slices from one patient to both training "
        "and test sets causes severe data leakage, leading to artificially inflated accuracy that fails catastrophically on unseen clinical patients.",
        prefix="1. Slice-Wise Data Leakage:"
    )
    add_bullet(
        "Inconsistencies between training-time preprocessing and deployment-time serving create train-test skew, "
        "causing deep learning models to fail when processing raw scans uploaded by clinicians.",
        prefix="2. Preprocessing Discrepancies:"
    )
    add_bullet(
        "Standard deep neural networks output probability scores without highlighting which anatomical regions "
        "influenced the prediction, limiting clinician trust and clinical adoptability.",
        prefix="3. Lack of Model Interpretability:"
    )

    # =========================================================
    # SECTION 4: OBJECTIVES OF THE PROJECT
    # =========================================================
    add_h1("4. Objectives of the Project")
    add_bullet("Develop a robust, reusable 13-stage modular preprocessing engine for structural MRI scans.", prefix="1.")
    add_bullet("Construct a strictly patient-wise partitioned supervised dataset from the OASIS-1 cohort to prevent data leakage.", prefix="2.")
    add_bullet("Implement a 2D EfficientNet-B0 deep neural network tailored for 224 × 224 2D MRI slice classification.", prefix="3.")
    add_bullet("Output a multi-class probability distribution across four CDR categories (CDR 0, 0.5, 1, and 2).", prefix="4.")
    add_bullet("Aggregate slice-level probability vectors to patient-level probability distributions via Mean Probability Aggregation.", prefix="5.")
    add_bullet("Derive an aggregated overall dementia-related classification probability from non-zero CDR classes.", prefix="6.")
    add_bullet("Rigorously evaluate model performance using multi-class medical machine learning metrics (ROC-AUC, F1-Score, Confusion Matrix).", prefix="7.")
    add_bullet("Incorporate 2D Grad-CAM / Grad-CAM++ Explainable AI for spatial feature attribution visualization.", prefix="8.")
    add_bullet("Integrate the preprocessing pipeline, model inference, and XAI heatmaps into a FastAPI + React web prototype.", prefix="9.")
    add_bullet("Establish a permanent, reproducible research data pipeline for future comparative AI studies (2D ResNet-18, Vision Transformers, SSL).", prefix="10.")

    # =========================================================
    # SECTION 5: METHODOLOGY
    # =========================================================
    add_h1("5. Methodology")
    add_p(
        "The project architecture follows Clean Architecture (Domain-Driven Design) principles, enforcing strict separation "
        "of concerns across modular packages: dataset management, preprocessing core, model registry, training engine, "
        "inference serving, backend REST API, and frontend web UI."
    )

    add_h2("5.1 Preprocessing Engine & Patient-Wise Data Splitting")
    add_p(
        "The preprocessing engine processes raw volumetric Analyze 7.5 scans from PROCESSED/MPRAGE/SUBJ_111/ through 13 sequential, "
        "deterministic operations: dataset discovery, metadata parsing, volume validation, spatial RAS reorientation, intensity Z-score "
        "normalization, brain bounding-box cropping, 2D slice extraction across Axial, Coronal, and Sagittal planes, Shannon entropy quality filtering "
        "(H ≥ 1.5, tissue ratio ≥ 15%), 224 × 224 slice resizing, PyTorch tensor serialization, and metadata logging."
    )
    add_p(
        "Supervised training utilizes the 234 labelled OASIS-1 subjects. Data splitting is enforced strictly at the subject level "
        "(80% Train / 187 patients, 10% Validation / 23 patients, 10% Test / 24 patients). Every slice belonging to Patient A "
        "remains exclusively within Patient A's designated split, guaranteeing zero slice-wise data leakage. The 179 unlabelled "
        "young control subjects are preserved in unlabelled_manifest.csv for self-supervised pre-training."
    )

    add_h2("5.2 2D EfficientNet-B0 Architecture & Slice Aggregation")
    add_p(
        "The primary deep-learning pipeline utilizes 2D EfficientNet-B0 operating on standardized 224 × 224 2D MRI slices (Axial plane default). "
        "To reduce computational overhead and accelerate training throughput while preserving transfer learning capabilities, ImageNet pre-trained "
        "weights are adapted for single-channel grayscale MRI slice inputs. The final classification head outputs a 4-class softmax "
        "probability vector P_slice = [p0, p0.5, p1, p2]."
    )
    add_p(
        "Slice-level probabilities across all valid slices of a patient are aggregated to the patient level via Mean Probability Aggregation:"
    )
    add_p("P_patient = Mean({P_slice, i}) = [p0, p0.5, p1, p2]")
    add_p(
        "An overall dementia-related classification probability is derived by aggregating non-zero CDR probabilities: "
        "P(Dementia) = P(CDR=0.5) + P(CDR=1.0) + P(CDR=2.0)."
    )

    add_h2("5.3 2D Grad-CAM Explainable AI")
    add_p(
        "Gradient-Weighted Class Activation Mapping (2D Grad-CAM / Grad-CAM++) computes the gradients of the target CDR class score "
        "with respect to feature maps in the final 2D convolutional layer of EfficientNet-B0, generating spatial heatmap overlays "
        "highlighting model feature attention on representative 2D MRI slices."
    )

    add_h2("5.4 Web Decision-Support Architecture")
    add_p(
        "The trained inference engine and XAI module are deployed as a web application prototype. The FastAPI backend receives uploaded "
        "MRI scans (.hdr/.img or .nii/.nii.gz), executes the shared preprocessing pipeline, computes slice predictions and patient mean "
        "aggregation, generates 2D Grad-CAM heatmaps, and serves JSON results to a React/Next.js dashboard for interactive radiological review."
    )

    # Architecture Table
    table_data = [
        ["System Module", "Core Components & Artifacts", "Responsibility & Data Flow"],
        ["Dataset Scanner & Splitter", "scanner.py, split_generator.py, manifests/", "Discovers 413 patients, parses metadata, generates patient-wise 80/10/10 splits."],
        ["Preprocessing Core", "loader.py, normalizer.py, cropper.py, resizer.py", "Executes 13-stage pipeline; generates 224x224 2D slices and 128x128x128 volumes."],
        ["PyTorch Data Pipeline", "torch_dataset.py, PyTorch DataLoader", "Feeds 2D [1,224,224] slice tensors (Axial default) to GPU training loop."],
        ["Deep Learning Model", "2D EfficientNet-B0 (models/cnn_2d/)", "Computes 4-class CDR probability distributions for valid 2D slices."],
        ["Slice Aggregation Engine", "trainer_2d.py, aggregate_slice_probabilities", "Aggregates slice probability vectors to patient level via Mean Probability Aggregation."],
        ["Explainable AI (XAI)", "2D Grad-CAM / Grad-CAM++ engine", "Generates 2D spatial heatmap overlays highlighting model feature attention on slices."],
        ["Inference Engine", "inference_preprocessor.py, engine.py", "Executes single-volume 2D slice extraction & prediction for uploaded MRI scans."],
        ["Backend REST API", "FastAPI (backend/app/main.py)", "Exposes /predict, /explain, and /report endpoints for clinical web UI."],
        ["Frontend Web App", "React / Next.js Dashboard", "Provides MRI drag-and-drop upload, 2D viewer, prediction scores, and PDF report."]
    ]

    t = doc.add_table(rows=len(table_data), cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    for row_idx, row in enumerate(table_data):
        for col_idx, text in enumerate(row):
            cell = t.cell(row_idx, col_idx)
            cell.text = text
            p_cell = cell.paragraphs[0]
            p_cell.paragraph_format.space_after = Pt(0)
            p_cell.paragraph_format.line_spacing = 1.15
            run_cell = p_cell.runs[0]

            if row_idx == 0:
                set_cell_background(cell, "1E3A8A") # Navy Blue
                run_cell.font.name = 'Arial'
                run_cell.font.bold = True
                run_cell.font.size = Pt(10)
                run_cell.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                if row_idx % 2 == 1:
                    set_cell_background(cell, "F8FAFC")
                else:
                    set_cell_background(cell, "FFFFFF")
                run_cell.font.name = 'Times New Roman'
                run_cell.font.size = Pt(10)
                run_cell.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                p_cell.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # =========================================================
    # SECTION 6: REFERENCES
    # =========================================================
    add_h1("6. References")
    refs = [
        "Marcus, D. S., et al. (2007). 'Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults.' Journal of Cognitive Neuroscience, 19(9), 1498–1507.",
        "Morris, J. C. (1993). 'The Clinical Dementia Rating (CDR): current version and scoring rules.' Neurology, 43(11), 2412–2414.",
        "Tan, M., & Le, Q. (2019). 'EfficientNet: Rethinking model scaling for convolutional neural networks.' In International Conference on Machine Learning (pp. 6105–6114).",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). 'Deep residual learning for image recognition.' In IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778).",
        "Selvaraju, R. R., et al. (2017). 'Grad-CAM: Visual explanations from deep networks via gradient-based localization.' In IEEE International Conference on Computer Vision (pp. 618–626).",
        "Jack, C. R., et al. (2018). 'NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease.' Alzheimer's & Dementia, 14(4), 535–562.",
        "Tustison, N. J., et al. (2010). 'N4ITK: improved N4 bias field correction.' IEEE Transactions on Medical Imaging, 29(6), 1310–1320.",
        "Ronneberger, O., Fischer, P., & Brox, T. (2015). 'U-net: Convolutional networks for biomedical image segmentation.' In MICCAI (pp. 234–241). Springer.",
        "Dosovitskiy, A., et al. (2020). 'An image is worth 16x16 words: Transformers for image recognition at scale.' arXiv preprint arXiv:2010.11929.",
        "Cardoso, M. J., et al. (2022). 'MONAI: An open-source framework for deep learning in healthcare.' arXiv preprint arXiv:2211.02701."
    ]
    for r in refs:
        add_bullet(r, prefix="[Ref]")

    doc.save(output_filename)
    print(f"Successfully generated college-formatted DOCX synopsis: {output_filename}")


def generate_college_odt_synopsis(output_filename="Alzheimers_Dementia_Stage_Classification_Synopsis.odt"):
    """Generates 100% valid ODT file following exact college section hierarchy."""
    mimetype_content = b"application/vnd.oasis.opendocument.text"

    manifest_xml = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
  <manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="application/vnd.oasis.opendocument.text"/>
  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

    meta_xml = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/" office:version="1.2">
  <office:meta>
    <dc:title>Explainable Deep Learning for Alzheimer's-Related Dementia Stage Classification Using Structural MRI</dc:title>
    <dc:creator>Academic Technical Researcher</dc:creator>
  </office:meta>
</office:document-meta>"""

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

    content_header = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" office:version="1.2">
  <office:automatic-styles>
    <style:style style:name="DocTitle" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center" fo:space-before="1.5cm" fo:space-after="0.4cm"/>
      <style:text-properties style:font-name="Arial" fo:font-size="20pt" fo:font-weight="bold" fo:color="#1e3a8a"/>
    </style:style>
    <style:style style:name="DocSubtitle" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center" fo:space-before="0.2cm" fo:space-after="1.0cm"/>
      <style:text-properties style:font-name="Arial" fo:font-size="12pt" fo:font-style="italic" fo:color="#2563eb"/>
    </style:style>
    <style:style style:name="DocMeta" style:family="paragraph">
      <style:paragraph-properties fo:text-align="center" fo:space-before="0.2cm" fo:space-after="0.15cm"/>
      <style:text-properties style:font-name="Arial" fo:font-size="10.5pt" fo:color="#475569"/>
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

    def escape_xml(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

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

    body_xml = []

    # Title Page
    body_xml.append(xml_p("Explainable Deep Learning for Alzheimer's-Related Dementia Stage Classification Using Structural MRI", "DocTitle"))
    body_xml.append(xml_p("An OASIS-1 Based Medical Imaging and Web-Based Decision Support System", "DocSubtitle"))
    body_xml.append(xml_p("MINOR PROJECT SYNOPSIS SUBMISSION", "DocMeta"))
    body_xml.append(xml_p("Submitted By (Team Members & Registration Numbers):", "DocMeta"))
    body_xml.append(xml_p("Spandan Chatterjee (Reg. No. 230905488) – Department of Computer Science & Engineering", "DocMeta"))
    body_xml.append(xml_p("Sambit Manna (Reg. No. 230905181) – Department of Computer Science & Engineering", "DocMeta"))
    body_xml.append(xml_p("Guru Asrith Nakka (Reg. No. 230905182) – Department of Computer Science & Engineering", "DocMeta"))
    body_xml.append(xml_p("Tanishque Srivastava (Reg. No. 220905123) – Department of Computer Science & Engineering", "DocMeta"))
    body_xml.append(xml_p("Minor Specialization: Artificial Intelligence & Machine Learning", "DocMeta"))
    body_xml.append(xml_p("Department Offering: Department of Computer Science & Engineering", "DocMeta"))
    body_xml.append(xml_p("Institution: Manipal Institute of Technology (MIT), Manipal", "DocMeta"))
    body_xml.append(xml_p("Faculty Guide / Mentor: [Faculty Guide Name]", "DocMeta"))
    body_xml.append(xml_p("Date of Submission: 15th March 2026", "DocMeta"))

    body_xml.append('      <text:p text:style-name="PageBreakPara"/>\n')

    # Section 1: Introduction
    body_xml.append(xml_h1("1. Introduction"))
    body_xml.append(xml_p(
        "Alzheimer's disease (AD) represents the primary cause of neurodegenerative dementia globally, characterized "
        "by progressive synaptic disruption, neuronal loss, and macrostructural brain atrophy. As populations age globally, "
        "the prevalence of Alzheimer's disease and related dementias is projected to affect over 130 million individuals by 2050, "
        "imposing an unprecedented socio-economic burden on healthcare systems worldwide. Early discrimination of dementia "
        "severity stages is paramount for timely therapeutic intervention, patient management, and clinical trial stratification."
    ))
    body_xml.append(xml_p(
        "In clinical practice, structural Magnetic Resonance Imaging (sMRI) is the non-invasive imaging modality of choice "
        "for evaluating macrostructural brain alterations associated with neurodegeneration. T1-weighted sMRI enables the "
        "visualization of anatomical hallmarks such as progressive hippocampal atrophy, entorhinal cortical thinning, and "
        "compensatory ventricular enlargement. However, manually inspecting hundreds of 2D MRI slices per patient is time-consuming, "
        "subject to inter-observer variability, and prone to missing subtle early-stage structural changes."
    ))
    body_xml.append(xml_p(
        "To address these challenges, this project presents an explainable, 2D slice-based deep-learning framework for four-class "
        "Alzheimer's-related dementia stage classification using structural sMRI scans from the Open Access Series of Imaging Studies "
        "(OASIS-1) cross-sectional cohort. The primary goal is to classify sMRI representations into Clinical Dementia Rating (CDR) "
        "severity stages: Cognitively Normal (CDR 0), Very Mild Dementia (CDR 0.5), Mild Dementia (CDR 1), and Moderate Dementia (CDR 2)."
    ))
    body_xml.append(xml_p(
        "To maximize computational efficiency, reduce GPU memory requirements, and accelerate model training while retaining "
        "critical anatomical features, the primary deep-learning pipeline utilizes 2D EfficientNet-B0 operating on standardized "
        "2D MRI slices (224 × 224) extracted across the Axial anatomical plane. Predictions generated at the slice level are "
        "aggregated to the patient level via Mean Probability Aggregation. Furthermore, Gradient-Weighted Class Activation Mapping "
        "(2D Grad-CAM / Grad-CAM++) is integrated to visualize salient anatomical features driving model predictions. Finally, "
        "the preprocessing engine, 2D inference model, and XAI visualizations are unified into a web-based clinical decision-support "
        "prototype built with FastAPI and React/Next.js."
    ))

    # Section 2: Literature Review (BRIEF)
    body_xml.append(xml_h1("2. Literature Review"))
    body_xml.append(xml_p(
        "Neuroimaging-based computer-aided diagnosis of dementia has evolved significantly with advances in machine learning "
        "and deep neural networks. Marcus et al. (2007) introduced the Open Access Series of Imaging Studies (OASIS-1) cross-sectional "
        "cohort, establishing a benchmark dataset of T1-weighted sMRI scans across non-demented and demented individuals aged 18 to 96. "
        "Morris (1993) formalized the Clinical Dementia Rating (CDR) scale, providing a validated clinical scoring system across six "
        "functional domains to grade dementia severity as Normal (CDR 0), Very Mild (CDR 0.5), Mild (CDR 1), and Moderate (CDR 2)."
    ))
    body_xml.append(xml_p(
        "In deep learning literature, Convolutional Neural Networks (CNNs) have shown strong capability in extracting volumetric "
        "and spatial biomarkers. He et al. (2016) introduced Deep Residual Learning (ResNet), mitigating vanishing gradient degradation "
        "in deep networks. Tan & Le (2019) developed EfficientNet, demonstrating that compound scaling of network depth, width, "
        "and resolution achieves superior feature representation efficiency with significantly fewer parameters compared to standard CNNs. "
        "For model interpretability, Selvaraju et al. (2017) formulated Gradient-Weighted Class Activation Mapping (Grad-CAM), "
        "enabling visual localization of salient image regions influencing network predictions."
    ))
    body_xml.append(xml_p(
        "Recent clinical frameworks (Jack et al., 2018; Cardoso et al., 2022) emphasize the necessity of combining high-efficiency "
        "deep neural networks with rigorous data partitioning and explainability tools to ensure reproducible and interpretable "
        "decision-support systems in neuroimaging research."
    ))

    # Section 3: Problem Statement
    body_xml.append(xml_h1("3. Problem Statement"))
    body_xml.append(xml_p("Current medical imaging AI research for dementia stage classification faces three critical vulnerabilities:"))
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

    # Section 4: Objectives
    body_xml.append(xml_h1("4. Objectives of the Project"))
    body_xml.append(xml_bullet("Develop a robust, reusable 13-stage modular preprocessing engine for structural MRI scans.", "1."))
    body_xml.append(xml_bullet("Construct a strictly patient-wise partitioned supervised dataset from the OASIS-1 cohort to prevent data leakage.", "2."))
    body_xml.append(xml_bullet("Implement a 2D EfficientNet-B0 deep neural network tailored for 224 × 224 2D MRI slice classification.", "3."))
    body_xml.append(xml_bullet("Output a multi-class probability distribution across four CDR categories (CDR 0, 0.5, 1, and 2).", "4."))
    body_xml.append(xml_bullet("Aggregate slice-level probability vectors to patient-level probability distributions via Mean Probability Aggregation.", "5."))
    body_xml.append(xml_bullet("Derive an aggregated overall dementia-related classification probability from non-zero CDR classes.", "6."))
    body_xml.append(xml_bullet("Rigorously evaluate model performance using multi-class medical machine learning metrics (ROC-AUC, F1-Score, Confusion Matrix).", "7."))
    body_xml.append(xml_bullet("Incorporate 2D Grad-CAM / Grad-CAM++ Explainable AI for spatial feature attribution visualization.", "8."))
    body_xml.append(xml_bullet("Integrate the preprocessing pipeline, model inference, and XAI heatmaps into a FastAPI + React web prototype.", "9."))
    body_xml.append(xml_bullet("Establish a permanent, reproducible research data pipeline for future comparative AI studies (2D ResNet-18, Vision Transformers, SSL).", "10."))

    # Section 5: Methodology
    body_xml.append(xml_h1("5. Methodology"))
    body_xml.append(xml_p(
        "The project architecture follows Clean Architecture (Domain-Driven Design) principles, enforcing strict separation "
        "of concerns across modular packages: dataset management, preprocessing core, model registry, training engine, "
        "inference serving, backend REST API, and frontend web UI."
    ))

    body_xml.append(xml_h2("5.1 Preprocessing Engine & Patient-Wise Data Splitting"))
    body_xml.append(xml_p(
        "The preprocessing engine processes raw volumetric Analyze 7.5 scans from PROCESSED/MPRAGE/SUBJ_111/ through 13 sequential, "
        "deterministic operations: dataset discovery, metadata parsing, volume validation, spatial RAS reorientation, intensity Z-score "
        "normalization, brain bounding-box cropping, 2D slice extraction across Axial, Coronal, and Sagittal planes, Shannon entropy quality filtering "
        "(H ≥ 1.5, tissue ratio ≥ 15%), 224 × 224 slice resizing, PyTorch tensor serialization, and metadata logging."
    ))
    body_xml.append(xml_p(
        "Supervised training utilizes the 234 labelled OASIS-1 subjects. Data splitting is enforced strictly at the subject level "
        "(80% Train / 187 patients, 10% Validation / 23 patients, 10% Test / 24 patients). Every slice belonging to Patient A "
        "remains exclusively within Patient A's designated split, guaranteeing zero slice-wise data leakage. The 179 unlabelled "
        "young control subjects are preserved in unlabelled_manifest.csv for self-supervised pre-training."
    ))

    body_xml.append(xml_h2("5.2 2D EfficientNet-B0 Architecture & Slice Aggregation"))
    body_xml.append(xml_p(
        "The primary deep-learning pipeline utilizes 2D EfficientNet-B0 operating on standardized 224 × 224 2D MRI slices (Axial plane default). "
        "To reduce computational overhead and accelerate training throughput while preserving transfer learning capabilities, ImageNet pre-trained "
        "weights are adapted for single-channel grayscale MRI slice inputs. The final classification head outputs a 4-class softmax "
        "probability vector P_slice = [p0, p0.5, p1, p2]."
    ))
    body_xml.append(xml_p(
        "Slice-level probabilities across all valid slices of a patient are aggregated to the patient level via Mean Probability Aggregation: "
        "P_patient = Mean({P_slice, i}) = [p0, p0.5, p1, p2]. An overall dementia-related classification probability is derived by aggregating "
        "non-zero CDR probabilities: P(Dementia) = P(CDR=0.5) + P(CDR=1.0) + P(CDR=2.0)."
    ))

    body_xml.append(xml_h2("5.3 2D Grad-CAM Explainable AI"))
    body_xml.append(xml_p(
        "Gradient-Weighted Class Activation Mapping (2D Grad-CAM / Grad-CAM++) computes the gradients of the target CDR class score "
        "with respect to feature maps in the final 2D convolutional layer of EfficientNet-B0, generating spatial heatmap overlays "
        "highlighting model feature attention on representative 2D MRI slices."
    ))

    body_xml.append(xml_h2("5.4 Web Decision-Support Architecture"))
    body_xml.append(xml_p(
        "The trained inference engine and XAI module are deployed as a web application prototype. The FastAPI backend receives uploaded "
        "MRI scans (.hdr/.img or .nii/.nii.gz), executes the shared preprocessing pipeline, computes slice predictions and patient mean "
        "aggregation, generates 2D Grad-CAM heatmaps, and serves JSON results to a React/Next.js dashboard for interactive radiological review."
    ))

    # ODT Architecture Table
    table_xml = []
    table_xml.append('      <table:table table:name="ArchitectureTable">\n')
    table_xml.append('        <table:table-column table:number-columns-repeated="3"/>\n')
    table_xml.append('        <table:table-row>\n')
    for h in ["System Module", "Core Components & Artifacts", "Responsibility & Data Flow"]:
        table_xml.append(f'          <table:table-cell table:style-name="TableHeadCellProp"><text:p text:style-name="TableHeaderCell">{escape_xml(h)}</text:p></table:table-cell>\n')
    table_xml.append('        </table:table-row>\n')

    rows = [
        ("Dataset Scanner & Splitter", "scanner.py, split_generator.py, manifests/", "Discovers 413 patients, parses metadata, generates patient-wise 80/10/10 splits."),
        ("Preprocessing Core", "loader.py, normalizer.py, cropper.py, resizer.py", "Executes 13-stage pipeline; generates 224x224 2D slices and 128x128x128 volumes."),
        ("PyTorch Data Pipeline", "torch_dataset.py, PyTorch DataLoader", "Feeds 2D [1,224,224] slice tensors (Axial default) to GPU training loop."),
        ("Deep Learning Model", "2D EfficientNet-B0 (models/cnn_2d/)", "Computes 4-class CDR probability distributions for valid 2D slices."),
        ("Slice Aggregation Engine", "trainer_2d.py, aggregate_slice_probabilities", "Aggregates slice probability vectors to patient level via Mean Probability Aggregation."),
        ("Explainable AI (XAI)", "2D Grad-CAM / Grad-CAM++ engine", "Generates 2D spatial heatmap overlays highlighting model feature attention on slices."),
        ("Inference Engine", "inference_preprocessor.py, engine.py", "Executes single-volume 2D slice extraction & prediction for uploaded MRI scans."),
        ("Backend REST API", "FastAPI (backend/app/main.py)", "Exposes /predict, /explain, and /report endpoints for clinical web UI."),
        ("Frontend Web App", "React / Next.js Dashboard", "Provides MRI drag-and-drop upload, 2D viewer, prediction scores, and PDF report.")
    ]

    for m, c, r in rows:
        table_xml.append('        <table:table-row>\n')
        table_xml.append(f'          <table:table-cell table:style-name="TableCellProp"><text:p text:style-name="TableCellPara">{escape_xml(m)}</text:p></table:table-cell>\n')
        table_xml.append(f'          <table:table-cell table:style-name="TableCellProp"><text:p text:style-name="TableCellPara">{escape_xml(c)}</text:p></table:table-cell>\n')
        table_xml.append(f'          <table:table-cell table:style-name="TableCellProp"><text:p text:style-name="TableCellPara">{escape_xml(r)}</text:p></table:table-cell>\n')
        table_xml.append('        </table:table-row>\n')

    table_xml.append('      </table>:table>\n')
    body_xml.append("".join(table_xml))

    # Section 6: References
    body_xml.append(xml_h1("6. References"))
    refs = [
        "Marcus, D. S., et al. (2007). 'Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults.' Journal of Cognitive Neuroscience, 19(9), 1498–1507.",
        "Morris, J. C. (1993). 'The Clinical Dementia Rating (CDR): current version and scoring rules.' Neurology, 43(11), 2412–2414.",
        "Tan, M., & Le, Q. (2019). 'EfficientNet: Rethinking model scaling for convolutional neural networks.' In International Conference on Machine Learning (pp. 6105–6114).",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). 'Deep residual learning for image recognition.' In IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778).",
        "Selvaraju, R. R., et al. (2017). 'Grad-CAM: Visual explanations from deep networks via gradient-based localization.' In IEEE International Conference on Computer Vision (pp. 618–626).",
        "Jack, C. R., et al. (2018). 'NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease.' Alzheimer's & Dementia, 14(4), 535–562.",
        "Tustison, N. J., et al. (2010). 'N4ITK: improved N4 bias field correction.' IEEE Transactions on Medical Imaging, 29(6), 1310–1320.",
        "Ronneberger, O., Fischer, P., & Brox, T. (2015). 'U-net: Convolutional networks for biomedical image segmentation.' In MICCAI (pp. 234–241). Springer.",
        "Dosovitskiy, A., et al. (2020). 'An image is worth 16x16 words: Transformers for image recognition at scale.' arXiv preprint arXiv:2010.11929.",
        "Cardoso, M. J., et al. (2022). 'MONAI: An open-source framework for deep learning in healthcare.' arXiv preprint arXiv:2211.02701."
    ]
    for r in refs:
        body_xml.append(xml_bullet(r, "[Ref]"))

    full_content_xml = content_header + "".join(body_xml) + content_footer

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", mimetype_content, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.xml", manifest_xml.encode("utf-8"))
        zf.writestr("meta.xml", meta_xml.encode("utf-8"))
        zf.writestr("styles.xml", styles_xml.encode("utf-8"))
        zf.writestr("content.xml", full_content_xml.encode("utf-8"))

    print(f"Successfully generated college-formatted ODT synopsis: {output_filename}")


def export_docx_to_pdf(docx_filename="Alzheimers_Dementia_Stage_Classification_Synopsis.docx", pdf_filename="Alzheimers_Dementia_Stage_Classification_Synopsis.pdf"):
    """Converts DOCX to PDF using Word COM object on Windows."""
    try:
        import win32com.client
        docx_path = os.path.abspath(docx_filename)
        pdf_path = os.path.abspath(pdf_filename)

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        doc = word.Documents.Open(docx_path)
        # 17 = wdFormatPDF
        doc.SaveAs(pdf_path, FileFormat=17)
        doc.Close()
        word.Quit()
        print(f"Successfully exported PDF: {pdf_filename}")
    except Exception as e:
        print(f"Error converting DOCX to PDF via Word COM: {e}")


if __name__ == "__main__":
    generate_college_docx_synopsis()
    generate_college_odt_synopsis()
    export_docx_to_pdf()
