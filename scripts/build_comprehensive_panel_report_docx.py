import os
import sys
import docx
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_report():
    doc = docx.Document()

    # Configure 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Color Palette
    PRIMARY_COLOR = RGBColor(16, 44, 87)       # Deep Navy
    SECONDARY_COLOR = RGBColor(53, 95, 142)    # Steel Blue
    ACCENT_COLOR = RGBColor(180, 83, 9)        # Warm Amber
    TEXT_DARK = RGBColor(33, 37, 41)          # Charcoal Text
    BG_LIGHT = "F4F6F9"
    BG_HEADER = "102C57"

    def format_run(run, font_name="Arial", size_pt=10.5, bold=False, italic=False, color_rgb=TEXT_DARK):
        run.font.name = font_name
        run.font.size = Pt(size_pt)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color_rgb

    def add_p(text="", style='Normal', space_after=6, space_before=0, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT):
        p = doc.add_paragraph(text, style=style)
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.line_spacing = line_spacing
        return p

    def add_h1(text):
        doc.add_page_break()
        p = add_p(space_after=12, space_before=16)
        r = p.add_run(text)
        format_run(r, font_name="Arial", size_pt=17, bold=True, color_rgb=PRIMARY_COLOR)
        return p

    def add_h2(text):
        p = add_p(space_after=8, space_before=14)
        r = p.add_run(text)
        format_run(r, font_name="Arial", size_pt=13, bold=True, color_rgb=SECONDARY_COLOR)
        return p

    def add_h3(text):
        p = add_p(space_after=6, space_before=10)
        r = p.add_run(text)
        format_run(r, font_name="Arial", size_pt=11, bold=True, color_rgb=PRIMARY_COLOR)
        return p

    def set_cell_background(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    def add_code_box(ascii_str):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.4)
        set_cell_background(cell, "F4F6F9")
        set_cell_margins(cell, 80, 80, 120, 120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(ascii_str)
        format_run(r, font_name="Consolas", size_pt=7.5, color_rgb=PRIMARY_COLOR)
        add_p(space_after=4)

    def add_metrics_table(headers, data_rows):
        tbl = doc.add_table(rows=len(data_rows) + 1, cols=len(headers))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        col_widths = [Inches(2.5), Inches(1.3), Inches(1.3), Inches(1.3)]
        for c_idx, h_text in enumerate(headers):
            cell = tbl.rows[0].cells[c_idx]
            if c_idx < len(col_widths):
                cell.width = col_widths[c_idx]
            set_cell_background(cell, BG_HEADER)
            set_cell_margins(cell, 70, 70, 100, 100)
            r = cell.paragraphs[0].add_run(h_text)
            format_run(r, font_name="Arial", size_pt=8.5, bold=True, color_rgb=RGBColor(255, 255, 255))
        for r_idx, row_vals in enumerate(data_rows):
            row = tbl.rows[r_idx + 1]
            bg = BG_LIGHT if r_idx % 2 == 0 else "FFFFFF"
            for c_idx, val in enumerate(row_vals):
                cell = row.cells[c_idx]
                if c_idx < len(col_widths):
                    cell.width = col_widths[c_idx]
                set_cell_background(cell, bg)
                set_cell_margins(cell, 50, 50, 80, 80)
                r = cell.paragraphs[0].add_run(val)
                format_run(r, font_name="Arial", size_pt=8.0, bold=(c_idx == 0 or r_idx >= len(data_rows) - 2), color_rgb=TEXT_DARK)
        add_p(space_after=8)

    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    cov_p = add_p(space_after=18, space_before=15, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_cov_inst = cov_p.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nMANIPAL INSTITUTE OF TECHNOLOGY (MIT), MANIPAL\n")
    format_run(r_cov_inst, font_name="Arial", size_pt=12, bold=True, color_rgb=SECONDARY_COLOR)

    cov_title_p = add_p(space_after=12, space_before=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_title = cov_title_p.add_run("EXPLAINABLE DEEP LEARNING FOR ALZHEIMER'S-RELATED DEMENTIA STAGE CLASSIFICATION USING STRUCTURAL MRI\n(NEUROSCREEN AI)")
    format_run(r_title, font_name="Arial", size_pt=17, bold=True, color_rgb=PRIMARY_COLOR)

    cov_sub_p = add_p(space_after=18, space_before=5, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_sub = cov_sub_p.add_run("MAJOR PROJECT TECHNICAL REPORT & PANEL DEFENSE SUBMISSION\nFrom Foundation to Tri-Modal Calibrated Gating Architecture")
    format_run(r_sub, font_name="Arial", size_pt=11, bold=True, color_rgb=SECONDARY_COLOR)

    team_p = add_p(space_after=14, space_before=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_team_hdr = team_p.add_run("Submitted By (Team Members & Registration Numbers):\n")
    format_run(r_team_hdr, font_name="Arial", size_pt=10, bold=True, color_rgb=PRIMARY_COLOR)
    r_members = team_p.add_run(
        "Spandan Chatterjee (Reg. No. 230905488) – Department of Computer Science & Engineering\n"
        "Sambit Manna (Reg. No. 230905181) – Department of Computer Science & Engineering\n"
        "Guru Asrith Nakka (Reg. No. 230905182) – Department of Computer Science & Engineering"
    )
    format_run(r_members, font_name="Arial", size_pt=9.5, color_rgb=TEXT_DARK)

    affil_p = add_p(space_after=16, space_before=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_bar = affil_p.add_run("|\n")
    format_run(r_bar, font_name="Arial", size_pt=11, bold=True, color_rgb=SECONDARY_COLOR)
    r_affil = affil_p.add_run(
        "Minor Specialization: Artificial Intelligence & Machine Learning\n"
        "Department Offering: Department of Computer Science & Engineering\n"
        "Institution: Manipal Institute of Technology (MIT), Manipal\n"
    )
    format_run(r_affil, font_name="Arial", size_pt=9.5, color_rgb=TEXT_DARK)
    r_guide = affil_p.add_run("Faculty Guide: Shankar Biradar\n")
    format_run(r_guide, font_name="Arial", size_pt=10, bold=True, color_rgb=PRIMARY_COLOR)

    cov_box = doc.add_table(rows=4, cols=2)
    cov_box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cov_box.autofit = False
    
    meta_rows = [
        ("Project Domain:", "Medical Artificial Intelligence, Computer Vision & Clinical NLP"),
        ("Single Target Variable:", "Clinical Dementia Rating (CDR 0: CN | CDR 0.5: MCI | CDR ≥1: AD)"),
        ("Winning Architecture:", "Tri-Modal Calibrated Gating (Vision + Tabular + PubMedBERT) [80.00%]"),
        ("Benchmark Cohorts:", "OASIS-1 + ADNI Multi-Center 1.5T/3T Patient Cohorts (104 Patients)")
    ]
    for idx, (label, val) in enumerate(meta_rows):
        c0, c1 = cov_box.rows[idx].cells
        c0.width = Inches(2.3)
        c1.width = Inches(4.2)
        set_cell_background(c0, "EAEEF3")
        set_cell_background(c1, "F8F9FA")
        set_cell_margins(c0, 70, 70, 110, 110)
        set_cell_margins(c1, 70, 70, 110, 110)
        r0 = c0.paragraphs[0].add_run(label)
        format_run(r0, font_name="Arial", size_pt=9, bold=True, color_rgb=PRIMARY_COLOR)
        r1 = c1.paragraphs[0].add_run(val)
        format_run(r1, font_name="Arial", size_pt=9, color_rgb=TEXT_DARK)

    add_p(space_after=20)
    auth_p = add_p(space_after=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_auth = auth_p.add_run("Submitted in Partial Fulfillment of Bachelor of Technology Degree\nAcademic Year 2025–2026")
    format_run(r_auth, font_name="Arial", size_pt=10.5, bold=True, color_rgb=PRIMARY_COLOR)

    # -------------------------------------------------------------
    # CONTENT PAGE (TABLE OF CONTENTS)
    # -------------------------------------------------------------
    add_h1("CONTENT PAGE")
    add_p("TABLE OF CONTENTS", align=WD_ALIGN_PARAGRAPH.CENTER)

    toc_table = doc.add_table(rows=15, cols=3)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_headers = ["Section No.", "Title / Topic", "Page Reference"]
    for c_idx, h_text in enumerate(toc_headers):
        cell = toc_table.rows[0].cells[c_idx]
        set_cell_background(cell, BG_HEADER)
        set_cell_margins(cell, 100, 100, 140, 140)
        r = cell.paragraphs[0].add_run(h_text)
        format_run(r, font_name="Arial", size_pt=9.5, bold=True, color_rgb=RGBColor(255, 255, 255))

    toc_rows = [
        ("—", "ABSTRACT & EXECUTIVE SUMMARY", "ii"),
        ("1", "INTRODUCTION & CLINICAL CONTEXT", "1"),
        ("2", "MOTIVATION", "3"),
        ("3", "PROBLEM DEFINITION AND OBJECTIVES", "5"),
        ("4", "SCOPE OF THE PROJECT", "7"),
        ("5", "TIMELINE USING GANTT CHART", "9"),
        ("6", "TECHNICAL REQUIREMENTS (HARDWARE & SOFTWARE)", "11"),
        ("7", "DATASET ARCHITECTURE & MULTI-COHORT HARMONIZATION", "13"),
        ("8", "SYSTEM ARCHITECTURE & METHODOLOGY (TRI-MODAL GATING)", "16"),
        ("9", "EMPIRICAL BENCHMARKS & INDEPENDENT MODEL EVOLUTION", "19"),
        ("10", "CLINICAL SAFETY & ERROR ANALYSIS (ZERO CATASTROPHIC ERRORS)", "23"),
        ("11", "VISUAL EXPLAINABILITY (REAL MRI GRAD-CAM HEATMAPS)", "25"),
        ("12", "CLINICAL INFERENCE & FUTURE SCOPE", "27"),
        ("13", "REFERENCES", "29")
    ]

    for r_idx, row_data in enumerate(toc_rows):
        row = toc_table.rows[r_idx + 1]
        bg = BG_LIGHT if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 70, 70, 120, 120)
            r = cell.paragraphs[0].add_run(val)
            format_run(r, font_name="Arial", size_pt=9, bold=(c_idx < 2), color_rgb=TEXT_DARK)

    # -------------------------------------------------------------
    # ABSTRACT
    # -------------------------------------------------------------
    add_h1("ABSTRACT")
    add_p(
        "Alzheimer's Disease (AD) and its prodromal stage, Mild Cognitive Impairment (MCI), represent an unprecedented global healthcare challenge. "
        "Clinical diagnosis relies on structural neuroimaging (sMRI) evaluated alongside patient cognitive assessments such as the Mini-Mental State "
        "Examination (MMSE) and detailed clinical consultation histories. While conventional computer vision approaches attempt to diagnose dementia purely "
        "from Magnetic Resonance Imaging (MRI) voxel intensities, they systematically struggle at the transition boundary between healthy aging (Cognitively Normal, CN) "
        "and early MCI due to subtle morphological overlap and scanner contrast variance. Conversely, text-only clinical NLP models fail to detect focal anatomical "
        "biomarkers such as medial temporal lobe atrophy and lateral ventricular enlargement."
    )
    add_p(
        "This project presents NeuroScreen AI, an end-to-end explainable multimodal deep learning architecture designed for clinical decision support in Alzheimer's staging. "
        "The system standardizes multi-cohort 1.5T and 3.0T structural MRI scans from OASIS-1 and ADNI (spanning 320+ human subjects and 76,428 axial slices), enforcing "
        "a strict patient-wise isolation protocol with zero data leakage. Slices are dynamically screened via Shannon entropy thresholding (H >= 1.5) and tissue density filtering (>=15%). "
        "The winning architecture, Tri-Modal Calibrated Gating, harmonizes three distinct latent representations: (1) a 512-dimensional visual vector extracted via a Multi-Kernel "
        "ResNet-18 backbone with Class-Weighted Focal Loss (gamma=2.0); (2) a 6-dimensional tabular anchor vector encoding exact patient age, biological sex, and discrete MMSE thresholds via a multi-layer perceptron (MLP); "
        "and (3) a 768-dimensional clinical semantic vector extracted from standardized patient consultations using PubMedBERT with attention-masked mean token pooling."
    )
    add_p(
        "A parametric dynamic gating network dynamically arbitrates modality contributions on a per-patient basis (averaging 59.3% text, 39.9% vision, and 0.8% tabular anchor), "
        "followed by temperature-scaled entropy calibration (T=1.19). Evaluated on an unseen blind holdout test cohort of 50 patients (and cross-verified on 104 multi-cohort holdout patients), "
        "the model achieves an 80.00% overall accuracy (0.7662 Macro F1) with an exceptional clinical safety profile: 100% Precision on severe Alzheimer's Disease (0 false positive AD diagnoses) "
        "and zero catastrophic classification errors (zero healthy patients diagnosed with AD, and zero AD patients classified as healthy). Visual explainability is delivered via high-resolution Grad-CAM "
        "heatmaps on real patient MRI scans focused on bilateral hippocampal parenchymal reduction and ventricular dilation, providing transparent, interpretable guidance for attending radiologists."
    )

    # -------------------------------------------------------------
    # 1. INTRODUCTION
    # -------------------------------------------------------------
    add_h1("1. INTRODUCTION")
    add_p(
        "Alzheimer's Disease (AD) is a progressive, neurodegenerative disorder that accounts for 60% to 80% of all dementia cases worldwide. "
        "Pathophysiologically, the disease is characterized by the accumulation of extracellular amyloid-beta plaques and intracellular hyperphosphorylated tau neurofibrillary tangles, "
        "which trigger progressive synaptic dysfunction, microglial neuroinflammation, and widespread neuronal apoptosis. Macroscopically, this progressive cellular loss results in "
        "severe cortical thinning, prominent atrophy of the medial temporal lobe (MTL) and hippocampus, and compensatory enlargement of the lateral ventricles."
    )
    add_p(
        "According to the World Health Organization (WHO), over 55 million individuals currently live with dementia globally, a figure projected to triple to 139 million by 2050, "
        "imposing an annual economic burden exceeding $1.3 trillion USD. Because disease-modifying therapies (such as anti-amyloid monoclonal antibodies) exhibit efficacy primarily "
        "when administered during early prodromal stages, timely and accurate staging is paramount."
    )
    add_p(
        "To establish rigorous clinical utility, this project standardizes its target variable according to the gold-standard Clinical Dementia Rating (CDR) scale:\n"
        "• Cognitively Normal (CN / CDR = 0.0): Intact memory, preserved functional independence, age-appropriate brain volume.\n"
        "• Mild Cognitive Impairment (MCI / CDR = 0.5): Mild memory deficits, preserved basic daily functioning, subtle hippocampal volume reduction.\n"
        "• Alzheimer's Disease (AD / CDR ≥ 1.0): Disabling cognitive impairment, loss of independence, pronounced ventricular and hippocampal atrophy."
    )

    # -------------------------------------------------------------
    # 2. MOTIVATION
    # -------------------------------------------------------------
    add_h1("2. MOTIVATION")
    add_p(
        "In modern hospital neurology departments, clinicians face a profound diagnostic dilemma when evaluating elderly patients presenting with memory complaints:"
    )
    add_p(
        "1. The Failure of Unimodal Computer Vision: Conventional deep learning models process 2D or 3D MRI scans in complete isolation. However, normal biological aging "
        "causes physiological brain volume reduction that closely mimics the early neurodegenerative atrophy of Mild Cognitive Impairment (MCI). An 80-year-old healthy individual "
        "and a 70-year-old in early MCI frequently exhibit nearly identical structural brain volumes. Consequently, pure vision models suffer from unacceptable false-positive rates "
        "on healthy elderly individuals (achieving only 47.3% precision on MCI in our ResNet-18 baseline)."
    )
    add_p(
        "2. The Limitations of Standalone Clinical NLP and Cognitive Tests: Standard cognitive screening instruments such as the Mini-Mental State Examination (MMSE, scored 0–30) "
        "are fast and non-invasive, but highly sensitive to patient educational attainment, socioeconomic background, language fluency, and transient anxiety. Moreover, cognitive scores "
        "cannot localize anatomical lesions or quantify physical disease acceleration (e.g., ventricular expansion rate)."
    )
    add_p(
        "3. The Multimodal Decision-Support Vision: Real radiologists and memory clinic specialists never diagnose a patient solely from an MRI scan without reading clinical history, "
        "nor do they prescribe dementia medications based solely on an MMSE score without structural imaging. NeuroScreen AI is motivated by the clinical necessity to replicate this "
        "multidisciplinary diagnostic process—fusing structural neuroimaging with cognitive intake narratives and discrete tabular cutoffs within a unified, transparent AI framework."
    )

    # -------------------------------------------------------------
    # 3. PROBLEM DEFINITION AND OBJECTIVES
    # -------------------------------------------------------------
    add_h1("3. PROBLEM DEFINITION AND OBJECTIVES")
    
    add_h2("3.1 Problem Definition")
    add_p(
        "Given a heterogeneous patient intake comprising: (1) a 3D structural T1-weighted MRI volume acquisition S ∈ R^(D x H x W); (2) structured patient demographic "
        "and cognitive tabular metrics M = [Age, Sex, MMSE, Has_MMSE]; and (3) an unstructured clinical consultation narrative T;\n"
        "The objective is to learn a parameterized multimodal decision-support function f_θ(S, M, T) that accurately predicts the patient's Clinical Dementia Rating "
        "y ∈ {CN (CDR 0.0), MCI (CDR 0.5), AD (CDR ≥ 1.0)}, while enforcing strict ordinal clinical safety (zero catastrophic misclassifications between CN and severe AD) "
        "and delivering spatial visual explainability via Gradient-Weighted Class Activation Mapping (Grad-CAM)."
    )

    add_h2("3.2 Specific Project Objectives")
    add_p(
        "To accomplish this problem definition, six measurable technical objectives were established:\n"
        "1. Multi-Cohort Data Harmonization: Harmonize multi-center 1.5T and 3.0T MRI acquisitions from OASIS-1 and ADNI repositories, enforcing a strict patient-wise isolation protocol (zero data leakage).\n"
        "2. Automated 2D Slice Processing Pipeline: Implement canonical RAS+ coordinate reorientation, percentile intensity standardization, and Shannon information entropy screening (H >= 1.5) to discard uninformative cranial slices.\n"
        "3. Advanced Computer Vision Backbone: Design and optimize a Multi-Kernel ResNet-18 feature extractor utilizing parallel multi-scale receptive fields (3x3, 5x5, 7x7) and Class-Weighted Focal Loss (gamma = 2.0) to address severe class imbalance.\n"
        "4. Domain-Adapted Clinical NLP Branch: Fine-tune PubMedBERT on clinical consultation narratives using Attention-Masked Mean Token Pooling rather than arbitrary [CLS] token pooling.\n"
        "5. Parametric Dynamic Gating & Temperature Calibration: Engineer a dynamic gating network that evaluates patient-specific modal confidence (learned weights g_vision, g_tabular, g_text) combined with temperature-scaled entropy calibration (T = 1.19).\n"
        "6. Interpretable Radiologist Decision Support: Validate model activations using Grad-CAM overlays on real patient MRI scans, ensuring decisions are grounded in genuine medial temporal lobe and ventricular pathology."
    )

    # -------------------------------------------------------------
    # 4. SCOPE OF THE PROJECT
    # -------------------------------------------------------------
    add_h1("4. SCOPE OF THE PROJECT")
    
    add_h2("4.1 In-Scope Technical Boundaries")
    add_p(
        "• Structural MRI Modality: High-resolution T1-weighted MPRAGE acquisitions from OASIS-1 (1.5T Siemens) and ADNI (3.0T multi-vendor).\n"
        "• Clinical Demographic Biomarkers: Patient Age, Biological Sex, and Mini-Mental State Examination (MMSE) scores.\n"
        "• Derived Morphometric Indices: Estimated Total Intracranial Volume (eTIV) and Normalized Whole Brain Volume (nWBV).\n"
        "• Target Classification Scheme: 3-Class Clinical Dementia Rating (CN vs MCI vs AD).\n"
        "• Explainability Integration: 2D slice-level Grad-CAM heatmaps highlighting hippocampal and ventricular atrophy.\n"
        "• Deployment Architecture: Decoupled FastAPI REST service and Radiologist Workstation interface."
    )

    add_h2("4.2 Out-of-Scope Technical Boundaries & Future Enhancements")
    add_p(
        "• Functional Neuroimaging: Functional MRI (fMRI) and Diffusion Tensor Imaging (DTI) are excluded due to prohibitive acquisition costs in primary care clinics.\n"
        "• Molecular & Nuclear Imaging: Positron Emission Tomography (PET) targeting amyloid-beta (e.g., Florbetapir) or tau tracers is excluded.\n"
        "• Invasive Biomarkers: Lumbar puncture cerebrospinal fluid (CSF) assays (Aβ42, t-tau, p-tau181) and blood-based plasma p-tau217 biomarkers are out of scope.\n"
        "• Genomic Sequencing: High-throughput genetic sequencing for Apolipoprotein E (APOE-ε4 allele) status is reserved for future multi-omic expansion."
    )

    # -------------------------------------------------------------
    # 5. TIMELINE USING GANTT CHART
    # -------------------------------------------------------------
    add_h1("5. TIMELINE USING GANTT CHART")
    add_p(
        "The project was executed across an 8-month intensive engineering timeline spanning August 2025 through April 2026. "
        "The timeline encompassed literature survey, multi-cohort data curation, unimodal baseline benchmarking, clinical NLP adaptation, "
        "multimodal gating engineering, temperature calibration, and explainability validation."
    )

    # Embed Gantt Chart
    gantt_path = os.path.join("reports", "project_gantt_chart.png")
    if os.path.exists(gantt_path):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(gantt_path, width=Inches(6.4))
        cap_p = add_p("Figure 5.1: Comprehensive Project Execution Timeline & Milestone Gantt Chart (Aug 2025 – Apr 2026).", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(12)

    # Gantt Milestones Table
    add_h2("5.1 Project Milestone Breakdown")
    gantt_table = doc.add_table(rows=11, cols=4)
    gantt_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    g_headers = ["Phase No.", "Milestone Description", "Duration", "Key Deliverables & Outcomes"]
    for c_idx, h_text in enumerate(g_headers):
        cell = gantt_table.rows[0].cells[c_idx]
        set_cell_background(cell, BG_HEADER)
        set_cell_margins(cell, 90, 90, 120, 120)
        r = cell.paragraphs[0].add_run(h_text)
        format_run(r, font_name="Arial", size_pt=9, bold=True, color_rgb=RGBColor(255, 255, 255))

    milestones_data = [
        ("Phase 1", "Clinical Problem Formulation & Setup", "24 Days", "Target CDR definition, ethics review, literature survey."),
        ("Phase 2", "Multi-Cohort Data Collection", "31 Days", "Acquisition of OASIS-1 & ADNI 1.5T/3T raw volumes."),
        ("Phase 3", "3D MRI Preprocessing Pipeline", "30 Days", "RAS+ coordinate alignment, Shannon entropy (H>=1.5) filter."),
        ("Phase 4", "Pure Vision CNN Baselines", "36 Days", "ResNet-18, DenseNet-121, EfficientNet-B0 benchmarking."),
        ("Phase 5", "Multi-Kernel ResNet-18 & Focal Loss", "35 Days", "Multi-scale kernel front-end, Focal Loss (gamma=2.0) [78.00%]."),
        ("Phase 6", "Clinical Text Modeling & PubMedBERT", "45 Days", "Biomedical adaptation, attention mean-pooling [74.00%]."),
        ("Phase 7", "Dynamic Gating & Tri-Modal Fusion", "36 Days", "Cross-modal gating network, tabular anchor integration [80.00%]."),
        ("Phase 8", "Temperature Calibration & Benchmark", "28 Days", "Learned T=1.19 scaling, 104-patient cross-cohort test."),
        ("Phase 9", "Visual Explainability (Grad-CAM)", "24 Days", "Real MRI Grad-CAM heatmaps over hippocampal regions."),
        ("Phase 10", "Academic Report & Defense", "26 Days", "Documentation, 300 DPI visualization suite, viva defense guide.")
    ]

    for r_idx, row_vals in enumerate(milestones_data):
        row = gantt_table.rows[r_idx + 1]
        bg = BG_LIGHT if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 100, 100)
            r = cell.paragraphs[0].add_run(val)
            format_run(r, font_name="Arial", size_pt=8.5, bold=(c_idx == 0), color_rgb=TEXT_DARK)

    # -------------------------------------------------------------
    # 6. TECHNICAL REQUIREMENTS
    # -------------------------------------------------------------
    add_h1("6. TECHNICAL REQUIREMENTS")
    add_p(
        "To ensure end-to-end reproducibility, the computational specifications for development, training, and deployment are formally categorized below:"
    )

    add_h2("6.1 Hardware Specifications")
    
    hw_table = doc.add_table(rows=5, cols=4)
    hw_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hw_headers = ["Component", "Cloud Training Infrastructure", "Development Workstation", "Hospital Deployment Spec"]
    for c_idx, h_text in enumerate(hw_headers):
        cell = hw_table.rows[0].cells[c_idx]
        set_cell_background(cell, BG_HEADER)
        set_cell_margins(cell, 90, 90, 120, 120)
        r = cell.paragraphs[0].add_run(h_text)
        format_run(r, font_name="Arial", size_pt=9, bold=True, color_rgb=RGBColor(255, 255, 255))

    hw_data = [
        ("Processor (CPU)", "Intel Xeon @ 2.20 GHz (2 vCPUs)", "AMD Ryzen 7 / Intel Core i7 (8 Cores)", "Quad-Core Intel i5 / Xeon (Standard PACS)"),
        ("Graphics (GPU)", "NVIDIA Tesla T4 (16 GB GDDR6)", "NVIDIA RTX 3060 / 4060 (8 GB VRAM)", "CPU-only Inference (No discrete GPU needed)"),
        ("System Memory", "13.0 GB High-Throughput RAM", "16.0 GB RAM (32 GB recommended)", "8.0 GB RAM minimum"),
        ("Storage (Disk)", "100 GB Cloud NVMe SSD", "500 GB PCIe Gen4 NVMe SSD", "20 GB available PACS storage")
    ]
    for r_idx, row_vals in enumerate(hw_data):
        row = hw_table.rows[r_idx + 1]
        bg = BG_LIGHT if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 100, 100)
            r = cell.paragraphs[0].add_run(val)
            format_run(r, font_name="Arial", size_pt=8.5, bold=(c_idx == 0), color_rgb=TEXT_DARK)

    add_p(space_after=12)

    add_h2("6.2 Software & Framework Specifications")
    sw_table = doc.add_table(rows=9, cols=3)
    sw_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sw_headers = ["Software Layer", "Framework / Library & Version", "Exact Functional Role in NeuroScreen AI"]
    for c_idx, h_text in enumerate(sw_headers):
        cell = sw_table.rows[0].cells[c_idx]
        set_cell_background(cell, BG_HEADER)
        set_cell_margins(cell, 90, 90, 120, 120)
        r = cell.paragraphs[0].add_run(h_text)
        format_run(r, font_name="Arial", size_pt=9, bold=True, color_rgb=RGBColor(255, 255, 255))

    sw_data = [
        ("Operating System", "Ubuntu 22.04 LTS / Windows 11 64-bit", "Host execution environment for cloud training and workstation deployment."),
        ("Deep Learning Core", "PyTorch 2.1.0+cu118 & Torchvision 0.16.0", "GPU-accelerated tensor computation, autograd, CNN and MLP modeling."),
        ("Clinical NLP Engine", "Hugging Face Transformers 4.35.2", "PubMedBERT tokenizer, attention masks, mean-token pooling computation."),
        ("Medical Imaging I/O", "NiBabel 5.1.0", "3D NIfTI (.nii, .nii.gz) and Analyze (.hdr/.img) loading and RAS+ affine mapping."),
        ("Scientific Computing", "NumPy 1.24.3 & SciPy 1.11.4", "Array processing, percentile intensity standardization, Shannon entropy calculation."),
        ("Evaluation & Metrics", "Scikit-Learn 1.3.2", "Patient-wise stratified splitting, confusion matrices, classification reports."),
        ("Visualization Suite", "Matplotlib 3.8.2 & Seaborn 0.13.0", "300 DPI publication heatmaps, multi-class ROC curves, radar competency charts."),
        ("Inference & REST API", "FastAPI 0.104.1 & Uvicorn 0.24.0", "Production asynchronous microservices for hospital clinical decision support.")
    ]
    for r_idx, row_vals in enumerate(sw_data):
        row = sw_table.rows[r_idx + 1]
        bg = BG_LIGHT if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 100, 100)
            r = cell.paragraphs[0].add_run(val)
            format_run(r, font_name="Arial", size_pt=8.5, bold=(c_idx == 0), color_rgb=TEXT_DARK)

    # -------------------------------------------------------------
    # 7. DATASET ARCHITECTURE & HARMONIZATION
    # -------------------------------------------------------------
    add_h1("7. DATASET ARCHITECTURE & MULTI-COHORT HARMONIZATION")
    add_p(
        "To establish true generalizability across diverse clinical settings, our dataset harmonizes two of the premier neuroimaging research repositories in the world:"
    )
    add_p(
        "• Open Access Series of Imaging Studies (OASIS-1): Cross-sectional MRI acquisition cohort of 234 subjects aged 18 to 96 years, acquired on a 1.5-Tesla Siemens Vision scanner. "
        "Provides standardized clinical demographic records including Age, Biological Sex, Socioeconomic Status (SES), Years of Education (EDUC), Mini-Mental State Examination (MMSE), "
        "Estimated Total Intracranial Volume (eTIV), and Normalized Whole Brain Volume (nWBV)."
    )
    add_p(
        "• Alzheimer's Disease Neuroimaging Initiative (ADNI): Multi-center prospective study acquired across high-field 3.0-Tesla scanners (Siemens, Philips, GE Healthcare) "
        "spanning ADNI-1, ADNI-2, ADNI-3, and ADNI-4 cohorts. Introduces substantial real-world variance in magnetic field strength, radiofrequency coil configurations, voxel contrast, and spatial resolution."
    )
    add_p(
        "Zero Data Leakage Protocol: Grouping was strictly executed at the human subject level before slice extraction: "
        "Train_Patients ∩ Val_Patients = Ø, Train_Patients ∩ Test_Patients = Ø. Every test patient evaluated in this report is a 100% unseen biological individual."
    )

    # -------------------------------------------------------------
    # 8. SYSTEM ARCHITECTURE & METHODOLOGY
    # -------------------------------------------------------------
    add_h1("8. SYSTEM ARCHITECTURE & MULTIMODAL METHODOLOGY")
    add_p(
        "The winning architecture, Tri-Modal Calibrated Gating, fuses three distinct latent modalities through a parametric dynamic gating mechanism and post-hoc temperature calibration:"
    )
    add_p(
        "1. Vision Stream (512-dim): Multi-Kernel ResNet-18 replacing the initial 7x7 conv with parallel (3x3, 5x5, 7x7) convolutions. "
        "Global average pooling yields visual feature vector v ∈ R^512, projected via linear layer into a 256-dim subspace.\n"
        "2. Tabular Stream (6-dim): Encodes exact Age, Sex, integer MMSE (0–30), and missingness flag into a 2-layer MLP (m ∈ R^6), projected into a 256-dim subspace. "
        "Acts as a hard mathematical anchor enforcing critical clinical MMSE cutoffs.\n"
        "3. Clinical Text Stream (768-dim): Standardized clinical consultation narratives embedded via fine-tuned PubMedBERT using Attention-Masked Mean Token Pooling:\n"
        "   t = [∑ mask_i * e_i] / [∑ mask_i] ∈ R^768\n"
        "4. Dynamic Gating ('The Chooser'): Computes patient-specific modal confidence:\n"
        "   g = Softmax(W_g [v; m; t] + b_g) = [g_vision, g_tabular, g_text]\n"
        "5. Temperature Calibration: Final logits scaled by learned parameter T = 1.19 to prevent overconfident misclassifications on borderline MCI cases."
    )

    # Embed Gating Distribution Graph
    gating_img = os.path.join("reports", "graph_gating_weights_distribution.png")
    if os.path.exists(gating_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(gating_img, width=Inches(6.0))
        cap_p = add_p("Figure 8.1: Learned Dynamic Gating Distribution across Diagnostic Stages (Notice Text Dominance in Ambiguous MCI).", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 9. EMPIRICAL BENCHMARKS & EXPERIMENTAL RESULTS
    # -------------------------------------------------------------
    add_h1("9. EMPIRICAL BENCHMARKS & EXPERIMENTAL RESULTS")
    add_p(
        "All models were evaluated across 104 total unseen patients across two independent cohorts: "
        "Primary Holdout Test Set (50 Patients, 8,325 slices) and Secondary Independent Cohort (54 Patients, 9,361 slices)."
    )

    # Embed Clean Top-4 + Baseline Benchmark Bar Chart
    bar_img = os.path.join("reports", "graph_model_comparison_bar.png")
    if os.path.exists(bar_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(bar_img, width=Inches(6.3))
        cap_p = add_p("Figure 9.1: Benchmark Comparison of Top 4 Models vs Clinical Baseline (Primary Acc, Macro F1, Dual-Cohort Acc).", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(12)

    # Embed ROC Curves
    roc_img = os.path.join("reports", "graph_roc_pr_curves.png")
    if os.path.exists(roc_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(roc_img, width=Inches(6.3))
        cap_p = add_p("Figure 9.2: Multi-Class ROC Curves (AUC = 0.968 on Alzheimer's) and Precision-Recall Curves.", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(12)

    # Embed Radar Chart
    radar_img = os.path.join("reports", "graph_classwise_radar_metrics.png")
    if os.path.exists(radar_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(radar_img, width=Inches(4.6))
        cap_p = add_p("Figure 9.3: Diagnostic Competency Comparison across Precision, Recall, and Overall Accuracy.", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 9.4 INDEPENDENT MODEL EVALUATIONS (ASCENDING ORDER: BASELINE TO WINNER)
    # -------------------------------------------------------------
    add_h2("9.4 Independent Diagnostic Profiles & Architecture Evolution")
    add_p(
        "To rigorously examine each model's independent behavioral profile, we present the top 4 candidate architectures alongside the baseline EfficientNet-B0 model. "
        "Each architecture is analyzed in ascending order of clinical performance, detailing its architectural topology, full confusion matrix, classification metrics, "
        "and the specific mathematical and clinical improvements it introduced over preceding iterations."
    )

    # ---------------------------------------------------------
    # MODEL 1 (BASELINE): EFFICIENTNET-B0 (60.00%)
    # ---------------------------------------------------------
    add_h3("9.4.1 Model 1 (Baseline Reference): Pure Vision EfficientNet-B0 (Test Acc: 60.00%)")
    
    ascii_eff = (
        "+-------------------------------------------------------------+\n"
        "|                    PATIENT INPUT DATA                       |\n"
        "|          2D Axial Structural MRI Slice (224x224)            |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|               EFFICIENTNET-B0 CNN BACKBONE                  |\n"
        "|       * Standard 3x3 Conv Stem (32 channels)                |\n"
        "|       * 7x Mobile Inverted Bottleneck (MBConv1, MBConv6)    |\n"
        "|       * Squeeze-and-Excitation (SE) Channel Attention       |\n"
        "|       * Swish Non-Linear Activation Function                |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|               GLOBAL AVERAGE POOLING (GAP)                  |\n"
        "|           torch.nn.AdaptiveAvgPool2d((1, 1))                |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|             1D EMBEDDING TENSOR (1280-dim)                  |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|                LINEAR CLASSIFICATION HEAD                   |\n"
        "|          Linear(1280, 3) + CrossEntropyLoss                 |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "                 PREDICTED CLINICAL CDR STAGE\n"
        "          [ CN: 42.9% | MCI: 60.0% | AD: 100.0% Recall ]"
    )
    add_code_box(ascii_eff)

    eff_cm_img = os.path.join("reports", "cm_efficientnet_b0.png")
    if os.path.exists(eff_cm_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(eff_cm_img, width=Inches(4.4))
        cap_p = add_p("Figure 9.4: Confusion Matrix for Model 1 (Baseline EfficientNet-B0) [Accuracy: 60.00%].", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(10)

    add_metrics_table(
        ["Diagnostic Class", "Precision", "Recall (Sens.)", "F1-Score"],
        [
            ["Cognitively Normal (CN)", "0.7500", "0.4286", "0.5455"],
            ["Mild Impairment (MCI)", "0.2727", "0.6000", "0.3750"],
            ["Alzheimer's (AD)", "1.0000", "1.0000", "1.0000"],
            ["Overall Accuracy", "60.00% (30/50)", "Macro F1", "0.6125"]
        ]
    )

    add_p(
        "• What We Did & Architectural Rationale: EfficientNet-B0 was deployed as our modern 3-class pure vision baseline. "
        "We sought to determine whether inverted residual bottleneck blocks (MBConv) and Squeeze-and-Excitation channel attention could accurately classify dementia stages from 2D axial MRI slices.\n"
        "• Diagnostic Findings: The model achieved 60.00% overall test accuracy (30/50 patients). Notably, it attained 100% sensitivity on severe AD (12/12 cases caught), confirming that macroscopic ventricular enlargement is readily detected by CNNs.\n"
        "• Failure Mode & Limitations: Healthy Cognitively Normal recall collapsed to 42.86%, with 16 out of 28 healthy elderly individuals misclassified as MCI. "
        "Depthwise separable convolutions over-compressed subtle grayscale ventricle edges, making the network hypersensitive to non-pathological senescent brain variations.\n"
        "• Key Progression Insight: Established that while severe AD is easily identified, distinguishing healthy aging from early MCI requires clinical contextual anchoring."
    )

    # ---------------------------------------------------------
    # MODEL 2: PUBMEDBERT (74.00%)
    # ---------------------------------------------------------
    add_h3("9.4.2 Model 2: Standalone PubMedBERT Clinical NLP (Test Acc: 74.00%)")
    
    ascii_bert = (
        "+-------------------------------------------------------------+\n"
        "|                    PATIENT INPUT DATA                       |\n"
        "|        Physician Intake: Age, Sex, MMSE, Symptoms           |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|           CLINICAL REPORT SYNTHESIZER ENGINE                |\n"
        "|      Standardized Neurological Consultation Summary         |\n"
        "|   e.g., '74-year-old male with MMSE 24/30, exhibiting       |\n"
        "|          mild memory complaints and intact ADLs...'         |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|             PUBMEDBERT BIOMEDICAL TRANSFORMER               |\n"
        "|    * microsoft/Biomedical-PubMedBERT-base-uncased           |\n"
        "|    * 12-Layer Transformer Encoder, 12 Attention Heads       |\n"
        "|    * Pretrained on 14M PubMed abstracts & full text         |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|             ATTENTION-WEIGHTED MEAN POOLING                 |\n"
        "|        Context-aware aggregation over sequence tokens       |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|            1D TEXT EMBEDDING VECTOR (768-dim)               |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|              CLASSIFICATION MLP HEAD + DROPOUT              |\n"
        "|       Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(3)  |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "                 PREDICTED CLINICAL CDR STAGE\n"
        "           [ CN: 89.3% Recall | 100% AD Precision: 81.8% ]"
    )
    add_code_box(ascii_bert)

    bert_cm_img = os.path.join("reports", "cm_pubmedbert_text.png")
    if os.path.exists(bert_cm_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(bert_cm_img, width=Inches(4.4))
        cap_p = add_p("Figure 9.5: Confusion Matrix for Model 2 (Standalone PubMedBERT) [Accuracy: 74.00%].", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(10)

    add_metrics_table(
        ["Diagnostic Class", "Precision", "Recall (Sens.)", "F1-Score"],
        [
            ["Cognitively Normal (CN)", "0.7576", "0.8929", "0.8197"],
            ["Mild Impairment (MCI)", "0.5000", "0.3000", "0.3750"],
            ["Alzheimer's (AD)", "0.8182", "0.7500", "0.7826"],
            ["Overall Accuracy", "74.00% (37/50)", "Macro F1", "0.6591"]
        ]
    )

    add_p(
        "• What We Did & Architectural Rationale: We completely bypassed imaging voxels to measure how accurately clinical narratives alone predict dementia stages. "
        "Using domain-specific PubMedBERT (110M parameters) pre-trained on biomedical literature, clinical summaries were synthesized from cognitive scores, age, and intake history.\n"
        "• What Improvements Happened Over Model 1:\n"
        "  1. +14.0% Overall Accuracy Leap: Test accuracy jumped from 60.00% to 74.00% without seeing any MRI scans.\n"
        "  2. Solved False Positives on Healthy Controls: Healthy Cognitively Normal recall soared from 42.86% to 89.29% (25/28 normal subjects correctly classified), eliminating Model 1's false-alarm failure.\n"
        "• Failure Mode & Limitations: The model is blind to anatomical pathology. In early prodromal MCI where cognitive MMSE scores remain borderline intact (26-28/30), text alone produced poor sensitivity: only 30.00% recall on MCI (3/10 caught).\n"
        "• Key Progression Insight: Proved that text is the gold-standard anchor for recognizing healthy individuals, but cannot detect early morphometric brain collapse."
    )

    # ---------------------------------------------------------
    # MODEL 3: RESNET-18 ADVANCED VISION (78.00%)
    # ---------------------------------------------------------
    add_h3("9.4.3 Model 3: Multi-Kernel Focal ResNet-18 (Advanced Pure Vision) (Test Acc: 78.00%)")
    
    ascii_res = (
        "+-------------------------------------------------------------+\n"
        "|                    PATIENT INPUT DATA                       |\n"
        "|       Volumetric 3D MRI Volume (.nii / .hdr / .dcm)         |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|        STANDARDIZATION & QUALITY FILTERING PIPELINE         |\n"
        "|   * Canonical RAS+ Reorientation (NiBabel)                  |\n"
        "|   * Percentile Intensity Normalization [0.5, 99.5%]         |\n"
        "|   * Shannon Entropy Filtering (H >= 1.5)                     |\n"
        "|   * 224x224 Foreground Axial Slice Extraction               |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|         MULTI-SCALE RECEPTIVE FIELD RESNET-18               |\n"
        "|   * Parallel Multi-Kernel Stem (3x3, 5x5, 7x7)              |\n"
        "|   * Residual Additive Skip Connections [x + F(x)]           |\n"
        "|   * 4 Residual Stages (64, 128, 256, 512 channels)          |\n"
        "|   * Prevents Scanner Noise Overfitting (11.7M params)       |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|        GLOBAL AVERAGE POOLING + TENSOR FLATTENING           |\n"
        "|           torch.flatten(AdaptiveAvgPool2d(1), 1)            |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|               1D VISION VECTOR (512-dim)                    |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "+-------------------------------------------------------------+\n"
        "|             CLASS-BALANCED FOCAL LOSS HEAD                  |\n"
        "|       Focal Loss (gamma=2.0, alpha=[0.81, 0.85, 1.73])      |\n"
        "+------------------------------+------------------------------+\n"
        "                               |\n"
        "                               v\n"
        "                 PREDICTED CLINICAL CDR STAGE\n"
        "          [ MCI Recall: 90.0% | CN Precision: 95.0% ]"
    )
    add_code_box(ascii_res)

    vis_cm_img = os.path.join("reports", "cm_vision_baseline.png")
    if os.path.exists(vis_cm_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(vis_cm_img, width=Inches(4.4))
        cap_p = add_p("Figure 9.6: Confusion Matrix for Model 3 (Multi-Kernel ResNet-18) [Accuracy: 78.00%].", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(10)

    add_metrics_table(
        ["Diagnostic Class", "Precision", "Recall (Sens.)", "F1-Score"],
        [
            ["Cognitively Normal (CN)", "0.9500", "0.6786", "0.7917"],
            ["Mild Impairment (MCI)", "0.4737", "0.9000", "0.6207"],
            ["Alzheimer's (AD)", "1.0000", "0.9167", "0.9565"],
            ["Overall Accuracy", "78.00% (39/50)", "Macro F1", "0.7896"]
        ]
    )

    add_p(
        "• What We Did & Architectural Rationale: We overhauled our vision pipeline, abandoning over-parameterized networks (ResNet-50, Swin-T) for a lightweight ResNet-18 (11.7M parameters) "
        "enhanced with multi-scale receptive field kernels and class-balanced Focal Loss (gamma=2.0).\n"
        "• What Improvements Happened Over Models 1 & 2:\n"
        "  1. 90.0% MCI Recall Breakthrough: MCI sensitivity surged 3x compared to PubMedBERT (from 30.0% to 90.00%, 9/10 MCI patients detected), capturing early hippocampal volume loss.\n"
        "  2. High Precision on Healthy Normal: Precision on CN reached 95.00% (19/20).\n"
        "  3. Zero False Negatives on AD: 11 of 12 AD patients were caught (the remaining 1 predicted as MCI; 0 missed into CN).\n"
        "• Failure Mode & Limitations: 9 healthy normal patients were predicted as MCI due to visual ambiguity from non-pathological senescent ventricular dilation.\n"
        "• Key Progression Insight: Solved early anatomical detection on MRI, proving that fusing this vision expert with clinical language would eliminate false MCI alarms."
    )

    # ---------------------------------------------------------
    # MODEL 4: DUAL-MODAL DYNAMIC GATING (80.00%)
    # ---------------------------------------------------------
    add_h3("9.4.4 Model 4: Multimodal Dynamic Gating (Vision + PubMedBERT) (Test Acc: 80.00%)")
    
    ascii_dual = (
        "+---------------------------------------------------------------------------+\n"
        "|                            PATIENT INPUT DATA                             |\n"
        "+---------------------+-------------------------------+---------------------+\n"
        "                      |                               |\n"
        "        [ 3D MRI Volume (.nii) ]        [ Clinical Intake Note & Symptoms ]\n"
        "                      |                               |\n"
        "                      v                               v\n"
        "            2D Axial Preprocessing         PubMedBERT Transformer (768-dim)\n"
        "            (Ventricles & Hippocampus)     (Biomedical Abstract Pretrained)\n"
        "                      |                               |\n"
        "                      v                               |\n"
        "            ResNet-18 Vision Backbone                 |\n"
        "            (Multi-Kernel + GAP + Flatten)            |\n"
        "                      |                               |\n"
        "                      v                               v\n"
        "            1D Vision Vector (512)          1D Text Vector (768)\n"
        "                      |                               |\n"
        "                      +---------------+---------------+\n"
        "                                      |\n"
        "                                      v\n"
        "                      +---------------------------------+\n"
        "                      |      DYNAMIC GATING NETWORK     |\n"
        "                      |          ('The Chooser')        |\n"
        "                      |   g = Sigmoid( W*[v; t] + b )   |\n"
        "                      +---------------+-----------------+\n"
        "                                      |\n"
        "                      Learned Dynamic Prioritization:\n"
        "                      * W_vision = g\n"
        "                      * W_text   = (1 - g)\n"
        "                                      |\n"
        "                                      v\n"
        "                      +---------------------------------+\n"
        "                      |       BLENDED LATENT SPACE      |\n"
        "                      |   h = g*v_proj + (1-g)*t_proj   |\n"
        "                      +---------------+-----------------+\n"
        "                                      |\n"
        "                                      v\n"
        "                      +---------------------------------+\n"
        "                      |    TOP CLASSIFICATION HEAD      |\n"
        "                      |   Final CDR Stage (CN/MCI/AD)   |\n"
        "                      +---------------------------------+"
    )
    add_code_box(ascii_dual)

    dual_cm_img = os.path.join("reports", "cm_dualmodal_gated.png")
    if os.path.exists(dual_cm_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(dual_cm_img, width=Inches(4.4))
        cap_p = add_p("Figure 9.7: Confusion Matrix for Model 4 (Dynamic Gating: Vision + Text) [Accuracy: 80.00%].", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(10)

    add_metrics_table(
        ["Diagnostic Class", "Precision", "Recall (Sens.)", "F1-Score"],
        [
            ["Cognitively Normal (CN)", "0.8276", "0.8571", "0.8421"],
            ["Mild Impairment (MCI)", "0.5455", "0.6000", "0.5714"],
            ["Alzheimer's (AD)", "1.0000", "0.8333", "0.9091"],
            ["Overall Accuracy", "80.00% (40/50)", "Macro F1", "0.7742"]
        ]
    )

    add_p(
        "• What We Did & Architectural Rationale: Engineered an adaptive dynamic soft-gating mechanism ('The Chooser'). "
        "A gating sub-network evaluates cross-modal agreement and computes an instance-specific weight g, dynamically balancing vision (g) and language (1-g).\n"
        "• What Improvements Happened Over Model 3:\n"
        "  1. Reached the 80.00% Accuracy Benchmark: Overall test accuracy advanced to 80.00% (40/50 patients), with Macro F1 of 0.7742 and Weighted F1 of 0.8040.\n"
        "  2. Cut Normal False Alarms by More than Half: Slashed Model 3's false MCI alarms from 9 down to 4 cases, returning healthy CN recall to 85.71%.\n"
        "  3. Autonomous Delegation: Dynamically assigned ~75% vision weight for obvious macroscopic atrophy cases while shifting to ~60% text weight for ambiguous normal aging cases.\n"
        "• Failure Mode & Limitations: 1 case of severe AD slipped into CN; continuous clinical values (exact numerical MMSE score, age) were only parsed indirectly through text tokens.\n"
        "• Key Progression Insight: Demonstrated the power of dynamic gating, prompting the addition of an explicit numerical tabular anchor branch to achieve zero catastrophic AD errors."
    )

    # ---------------------------------------------------------
    # MODEL 5 (CHAMPION): TRI-MODAL CALIBRATED GATING (80.00% / 100% AD Precision)
    # ---------------------------------------------------------
    add_h3("9.4.5 Model 5 (Winning Model): Tri-Modal Calibrated Gating (Test Acc: 80.00% | 100% AD Precision)")
    
    ascii_tri = (
        "+----------------------------------------------------------------------------------------+\n"
        "|                                   PATIENT INPUT DATA                                   |\n"
        "+--------------+---------------------------+------------------------------+--------------+\n"
        "               |                           |                              |\n"
        " [ 3D MRI Volume (.nii) ]   [ Tabular Biomarkers: Age,    [ Clinical Consultation Note ]\n"
        "               |              Sex, MMSE, EDUC, nWBV ]                     |\n"
        "               v                           |                              v\n"
        "     Multi-Kernel ResNet-18                v                    PubMedBERT Transformer\n"
        "     (GAP + Flattening)             Tabular MLP Head            (Attention Mean Pooling)\n"
        "               |                       (2 Layers)                         |\n"
        "               v                           |                              v\n"
        "      1D Vision (512-dim)         1D Tabular (64-dim)            1D Text Vector (768-dim)\n"
        "               |                           |                              |\n"
        "               +---------------------------+------------------------------+\n"
        "                                           |\n"
        "                                           v\n"
        "                            Linear Shared Projection (256-dim)\n"
        "                                           |\n"
        "                                           v\n"
        "                       +---------------------------------------+\n"
        "                       |      TRI-MODAL DYNAMIC GATING         |\n"
        "                       |           ('The Chooser')             |\n"
        "                       |  [g_v, g_t, g_m] = Softmax(W*[v;t;m]) |\n"
        "                       +-------------------+-------------------+\n"
        "                                           |\n"
        "                        Context-Calibrated Latent Fusion:\n"
        "                     h = g_v*v_proj + g_t*t_proj + g_m*m_proj\n"
        "                                           |\n"
        "                                           v\n"
        "                       +---------------------------------------+\n"
        "                       |     TEMPERATURE SCALING CALIBRATION   |\n"
        "                       |             (T = 1.19)                |\n"
        "                       |     p = Softmax( logits / 1.19 )      |\n"
        "                       +-------------------+-------------------+\n"
        "                                           |\n"
        "                                           v\n"
        "                             PREDICTED CLINICAL CDR STAGE\n"
        "                     [ 100% Precision on Alzheimer's Disease ]\n"
        "                     [ ZERO False Positives on Severe AD     ]\n"
        "                     [ ZERO Catastrophic Normal-to-AD Errors ]"
    )
    add_code_box(ascii_tri)

    tri_cm_img = os.path.join("reports", "cm_trimodal_calibrated.png")
    if os.path.exists(tri_cm_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(tri_cm_img, width=Inches(4.4))
        cap_p = add_p("Figure 9.8: Confusion Matrix for Model 5 (Tri-Modal Calibrated Gating - Champion) [Accuracy: 80.00% | 100% AD Precision].", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(10)

    add_metrics_table(
        ["Diagnostic Class", "Precision", "Recall (Sens.)", "F1-Score"],
        [
            ["Cognitively Normal (CN)", "0.8276", "0.8571", "0.8421"],
            ["Mild Impairment (MCI)", "0.5000", "0.5000", "0.5000"],
            ["Alzheimer's (AD)", "1.0000", "0.9167", "0.9565"],
            ["Overall Accuracy", "80.00% (40/50)", "Macro F1", "0.7662"]
        ]
    )

    add_p(
        "• What We Did & Architectural Rationale: Developed our final champion architecture: an end-to-end 3-way Softmax Dynamic Gating network fusing (1) 512-dim Multi-Kernel ResNet-18 vision, "
        "(2) a dedicated 64-dim Tabular MLP processing hard numerical patient biomarkers (Age, Biological Sex, MMSE, EDUC, nWBV), and (3) 768-dim PubMedBERT, "
        "calibrated with post-hoc Temperature Scaling (T=1.19).\n"
        "• What Improvements Happened Over Model 4:\n"
        "  1. 100% Precision on Alzheimer's Disease: 11 out of 11 predicted AD cases were verified true AD cases (0 false positives). Exactly zero healthy or MCI patients were misdiagnosed with severe dementia.\n"
        "  2. Zero Catastrophic Clinical Errors: Both catastrophic error modes (CN -> AD and AD -> CN) are strictly ZERO.\n"
        "  3. Highest Dual-Cohort Combined Accuracy: Across 104 unseen test patients (combining Primary and Secondary holdout cohorts), this model achieved 77.88% combined accuracy (81/104 patients), outperforming every single evaluated baseline."
    )

    # ---------------------------------------------------------
    # 9.5 PROGRESSIVE EVOLUTION SYNTHESIS TABLE
    # ---------------------------------------------------------
    add_h2("9.5 Synthesis of Progressive Architectural Improvements")
    add_p(
        "The table below summarizes the progressive evolution across all five architectures, demonstrating how each model systematically resolved the limitations of its predecessor:"
    )

    synth_table = doc.add_table(rows=6, cols=5)
    synth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    synth_headers = ["Model Iteration", "Architecture Type", "Test Acc.", "Macro F1", "Key Clinical Improvement & Solved Failure Mode"]
    col_w = [Inches(1.2), Inches(1.3), Inches(0.9), Inches(0.8), Inches(2.2)]
    for c_idx, h_text in enumerate(synth_headers):
        cell = synth_table.rows[0].cells[c_idx]
        cell.width = col_w[c_idx]
        set_cell_background(cell, BG_HEADER)
        set_cell_margins(cell, 80, 80, 100, 100)
        r = cell.paragraphs[0].add_run(h_text)
        format_run(r, font_name="Arial", size_pt=8.5, bold=True, color_rgb=RGBColor(255, 255, 255))

    synth_rows = [
        ("Model 1 (Baseline)", "EfficientNet-B0 Vision", "60.00%", "0.6125", "First 3-class baseline. 100% AD sensitivity, but low CN recall (42.9%) due to slice contrast sensitivity."),
        ("Model 2", "Standalone PubMedBERT", "74.00%", "0.6591", "+14.0% accuracy jump. Solved CN false alarms (89.3% recall), but blind to internal brain atrophy (30% MCI recall)."),
        ("Model 3", "Multi-Kernel ResNet-18", "78.00%", "0.7896", "MCI recall surged to 90.0% (3x jump). 95% CN precision. Zero severe AD false negatives."),
        ("Model 4", "Dynamic Gating (Dual-Modal)", "80.00%", "0.7742", "Broke 80% threshold. Cut CN false alarms from 9 to 4 cases. Autonomous modality weight delegation."),
        ("Model 5 (Champion)", "Tri-Modal Calibrated Gating", "80.00%", "0.7662", "100% AD Precision (0 false alarms). Zero catastrophic errors. 77.88% combined accuracy across 104 patients.")
    ]
    for r_idx, r_data in enumerate(synth_rows):
        row = synth_table.rows[r_idx + 1]
        bg = BG_LIGHT if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = row.cells[c_idx]
            cell.width = col_w[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 80, 80)
            r = cell.paragraphs[0].add_run(val)
            format_run(r, font_name="Arial", size_pt=8.0, bold=(c_idx == 0 or r_idx == 4), color_rgb=TEXT_DARK)

    add_p(space_after=14)

    # -------------------------------------------------------------
    # 10. CLINICAL SAFETY & ERROR ANALYSIS
    # -------------------------------------------------------------
    add_h1("10. CLINICAL SAFETY & ERROR ANALYSIS")
    add_p(
        "In healthcare diagnostics, errors are not equal. Standard ML treats errors uniformly, but in medical practice:\n"
        "• Catastrophic False Positive (CN -> AD): Telling a healthy individual they have incurable dementia causes catastrophic trauma.\n"
        "• Catastrophic False Negative (AD -> CN): Reassuring an active Alzheimer's patient that they are healthy misses critical care windows."
    )
    add_p(
        "Empirical Safety Guarantee: In our winning Tri-Modal Calibrated Gating model, BOTH catastrophic error cells are EXACTLY ZERO: "
        "0 out of 28 healthy controls diagnosed with AD, and 0 out of 12 AD patients diagnosed as healthy. "
        "All misclassifications occur strictly between adjacent diagnostic stages (CN <-> MCI or MCI <-> AD), satisfying the clinical criterion of Graceful Ordinal Degradation."
    )

    # -------------------------------------------------------------
    # 11. VISUAL EXPLAINABILITY (REAL MRI GRAD-CAM)
    # -------------------------------------------------------------
    add_h1("11. VISUAL EXPLAINABILITY (REAL MRI GRAD-CAM)")
    add_p(
        "To provide actionable, transparent decision support for attending radiologists, NeuroScreen AI implements Gradient-Weighted Class Activation Mapping (Grad-CAM) "
        "targeting the final residual block of the Multi-Kernel ResNet-18 backbone."
    )

    # Embed Real MRI Grad-CAM Panel
    gradcam_img = os.path.join("reports", "gradcam_visual_explainability.png")
    if os.path.exists(gradcam_img):
        p_img = add_p(align=WD_ALIGN_PARAGRAPH.CENTER)
        p_img.add_run().add_picture(gradcam_img, width=Inches(6.2))
        cap_p = add_p("Figure 11.1: Clinical Grad-CAM Visual Heatmaps on Real Patient T1 MRI Scans across CN, MCI, and AD staging.", align=WD_ALIGN_PARAGRAPH.CENTER)
        cap_p.paragraph_format.space_after = Pt(12)

    add_p(
        "Clinical Anatomical Correlation:\n"
        "• Cognitively Normal (CN): Grad-CAM displays diffuse, low-intensity cortical background without focal hot-spots, confirming preserved parenchyma.\n"
        "• Mild Cognitive Impairment (MCI): Clear focal twin activations emerge over the Medial Temporal Lobe and anterior hippocampus, detecting early localized volume loss.\n"
        "• Alzheimer's Disease (AD): Intense, concentrated activations highlight profound bilateral hippocampal atrophy and periventricular tissue bordering severely dilated lateral ventricles."
    )

    # -------------------------------------------------------------
    # 12. CLINICAL INFERENCE & FUTURE SCOPE
    # -------------------------------------------------------------
    add_h1("12. CLINICAL INFERENCE & FUTURE SCOPE")
    add_p(
        "The production deployment architecture decouples model inference into a high-throughput FastAPI REST microservice. "
        "Clinicians upload a 3D MRI volume (.nii/.dcm) and input patient age, sex, and MMSE score. "
        "The pipeline executes automated RAS+ standardization, entropy slice filtering, clinical report synthesis, and Tri-Modal gated inference, "
        "returning predicted CDR stage, modality contribution weights, and real-time Grad-CAM overlays in under 1.2 seconds."
    )
    add_p(
        "Future Enhancements: (1) Prospective multi-center clinical validation in hospital PACS networks; (2) Longitudinal disease trajectory modeling to predict conversion from MCI to AD 24 months in advance; "
        "and (3) Integration of non-invasive plasma p-tau217 blood biomarkers into the tabular MLP stream."
    )

    # -------------------------------------------------------------
    # 13. REFERENCES
    # -------------------------------------------------------------
    add_h1("13. REFERENCES")
    refs = [
        "1. Jack, C. R., et al. 'NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease.' Alzheimer's & Dementia 14.4 (2018): 535-562.",
        "2. Marcus, D. S., et al. 'Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults.' Journal of Cognitive Neuroscience 19.9 (2007): 1498-1507.",
        "3. Mueller, S. G., et al. 'Ways toward an early diagnosis in Alzheimer's disease: The Alzheimer's Disease Neuroimaging Initiative (ADNI).' Alzheimer's & Dementia 1.1 (2005): 55-66.",
        "4. Gu, Y., et al. 'Domain-specific language model pretraining for biomedical natural language processing.' ACM Transactions on Information Systems 40.1 (2021): 1-23. (PubMedBERT)",
        "5. Selvaraju, R. R., et al. 'Grad-CAM: Visual explanations from deep networks via gradient-based localization.' IEEE International Conference on Computer Vision (ICCV), 2017.",
        "6. Lin, T. Y., et al. 'Focal loss for dense object detection.' IEEE International Conference on Computer Vision (ICCV), 2017.",
        "7. Guo, C., et al. 'On calibration of modern neural networks.' International Conference on Machine Learning (ICML). PMLR, 2017."
    ]
    for ref in refs:
        p = add_p(ref, space_after=6)
        format_run(p.runs[0], font_name="Arial", size_pt=9.5, color_rgb=TEXT_DARK)

    output_path = "reports/NeuroScreen_AI_Master_College_Technical_Report.docx"
    try:
        doc.save(output_path)
        print(f"✅ Master 25-30 Page College Technical Report successfully saved to: {output_path}")
    except PermissionError:
        fallback_path = "reports/NeuroScreen_AI_Master_College_Technical_Report_Updated.docx"
        doc.save(fallback_path)
        print(f"⚠️ Primary file was locked in Word. Saved successfully to: {fallback_path}")

if __name__ == "__main__":
    create_report()
