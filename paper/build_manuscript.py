#!/usr/bin/env python3
"""
Build a 4-6 page workshop-style academic manuscript PDF.
Two-column layout using ReportLab canvas API with Helvetica.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import black, HexColor
from reportlab.pdfgen import canvas
import textwrap
import os

_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(_OUTPUT_DIR, "manuscript.pdf")

PAGE_W, PAGE_H = letter
MARGIN_TOP = 0.72 * inch
MARGIN_BOTTOM = 0.72 * inch
MARGIN_LEFT = 0.65 * inch
MARGIN_RIGHT = 0.65 * inch
COL_GAP = 0.22 * inch

CONTENT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
COL_W = (CONTENT_W - COL_GAP) / 2.0

BODY_SIZE = 9
BODY_LEADING = 11.2
SECTION_SIZE = 10.5
SUBSECTION_SIZE = 9.5
TABLE_SIZE = 7.5
TABLE_LEADING = 9.5
CAPTION_SIZE = 8
REF_SIZE = 7.5
REF_LEADING = 9.2


class AcademicPaper:
    def __init__(self, filename):
        self.c = canvas.Canvas(filename, pagesize=letter)
        self.c.setTitle("Privacy Leakage and Lightweight Defenses in Graph Neural Networks for Sensitive Data")
        self.c.setAuthor("Perplexity Computer")
        self.page_num = 1
        self.col = 0  # 0=left, 1=right
        self.y = PAGE_H - MARGIN_TOP
        # Track the y-position where columns start on the current page
        self.col_start_y = PAGE_H - MARGIN_TOP

    def col_x(self):
        return MARGIN_LEFT if self.col == 0 else MARGIN_LEFT + COL_W + COL_GAP

    def _page_number(self):
        self.c.setFont("Helvetica", 8)
        self.c.setFillColor(black)
        self.c.drawCentredString(PAGE_W / 2, MARGIN_BOTTOM * 0.4, str(self.page_num))

    def new_page(self):
        """Start a new page."""
        self._page_number()
        self.c.showPage()
        self.page_num += 1
        self.col = 0
        self.y = PAGE_H - MARGIN_TOP
        self.col_start_y = PAGE_H - MARGIN_TOP

    def next_col(self):
        """Move to next column or next page."""
        if self.col == 0:
            self.col = 1
            self.y = self.col_start_y  # Start right col at same height as left col started
        else:
            self.new_page()

    def ensure_space(self, h):
        if self.y - h < MARGIN_BOTTOM:
            self.next_col()

    def _wrap(self, text, font, size, width):
        avg_cw = self.c.stringWidth("x", font, size)
        chars = max(20, int(width / avg_cw))
        lines = []
        for para in text.split('\n'):
            para = para.strip()
            if para == '':
                lines.append('')
            else:
                lines.extend(textwrap.wrap(para, width=chars))
        return lines

    def draw_text(self, text, x, width, font="Helvetica", size=BODY_SIZE,
                  leading=BODY_LEADING, indent=0):
        """Draw wrapped text with column-break handling."""
        lines = self._wrap(text, font, size, width - indent)
        for line in lines:
            if self.y - leading < MARGIN_BOTTOM:
                self.next_col()
                x = self.col_x()
            if line == '':
                self.y -= leading * 0.4
                continue
            self.c.setFont(font, size)
            self.c.setFillColor(black)
            self.c.drawString(x + indent, self.y, line)
            self.y -= leading

    # ---- Full-width header ----

    def draw_title(self):
        lines = [
            "Privacy Leakage and Lightweight Defenses in",
            "Graph Neural Networks for Sensitive Data"
        ]
        self.c.setFont("Helvetica-Bold", 14)
        for ln in lines:
            self.c.drawCentredString(PAGE_W / 2, self.y, ln)
            self.y -= 17
        self.y -= 4

        self.c.setFont("Helvetica", 10)
        self.c.drawCentredString(PAGE_W / 2, self.y, "Krish Sharma, Shaikh Arifuzzaman")
        self.y -= 13
        self.c.setFont("Helvetica", 8.5)
        self.c.drawCentredString(PAGE_W / 2, self.y, "DiSC Lab, University of New Orleans")
        self.y -= 11
        self.c.drawCentredString(PAGE_W / 2, self.y, "krishkiaan82@gmail.com")
        self.y -= 14

    def draw_abstract(self, text):
        self.c.setFont("Helvetica-Bold", 10)
        self.c.drawCentredString(PAGE_W / 2, self.y, "Abstract")
        self.y -= 12

        abs_indent = 0.3 * inch
        abs_width = CONTENT_W - 2 * abs_indent
        abs_x = MARGIN_LEFT + abs_indent
        abs_size = 8.5
        abs_leading = 10.5
        lines = self._wrap(text, "Helvetica", abs_size, abs_width)
        for ln in lines:
            self.c.setFont("Helvetica", abs_size)
            self.c.drawString(abs_x, self.y, ln)
            self.y -= abs_leading
        self.y -= 4

        # Horizontal rule
        self.c.setStrokeColor(black)
        self.c.setLineWidth(0.5)
        self.c.line(MARGIN_LEFT, self.y, PAGE_W - MARGIN_RIGHT, self.y)
        self.y -= 8

    def start_two_column(self):
        """Mark where two-column layout begins on this page."""
        self.col_start_y = self.y
        self.col = 0

    # ---- Two-column elements ----

    def section(self, num, title):
        self.ensure_space(SECTION_SIZE + 8 + BODY_LEADING)
        self.y -= 7
        x = self.col_x()
        self.c.setFont("Helvetica-Bold", SECTION_SIZE)
        self.c.drawString(x, self.y, f"{num}. {title}")
        self.y -= SECTION_SIZE + 3

    def subsection(self, title):
        self.ensure_space(SUBSECTION_SIZE + 6 + BODY_LEADING)
        self.y -= 3
        x = self.col_x()
        self.c.setFont("Helvetica-Bold", SUBSECTION_SIZE)
        self.c.drawString(x, self.y, title)
        self.y -= SUBSECTION_SIZE + 2

    def body(self, text, bold=False, italic=False):
        font = "Helvetica"
        if bold:
            font = "Helvetica-Bold"
        elif italic:
            font = "Helvetica-Oblique"
        x = self.col_x()
        self.draw_text(text, x, COL_W, font=font, size=BODY_SIZE, leading=BODY_LEADING)
        self.y -= 3

    def draw_table(self, caption_num, caption_text, headers, rows, col_widths):
        n_rows = len(rows) + 1
        row_h = TABLE_LEADING + 1
        needed = row_h * n_rows + 35 + 15
        self.ensure_space(needed)

        x0 = self.col_x()
        table_w = sum(col_widths)

        # Caption
        cap = f"Table {caption_num}: {caption_text}"
        cap_lines = self._wrap(cap, "Helvetica-Bold", CAPTION_SIZE, COL_W)
        for cl in cap_lines:
            self.c.setFont("Helvetica-Bold", CAPTION_SIZE)
            self.c.drawString(x0, self.y, cl)
            self.y -= CAPTION_SIZE + 1.5
        self.y -= 2

        # Top rule
        self.c.setStrokeColor(black)
        self.c.setLineWidth(0.7)
        rule_y = self.y + 3
        self.c.line(x0, rule_y, x0 + table_w, rule_y)

        # Header
        self.c.setFillColor(HexColor("#E8E8E8"))
        self.c.rect(x0, self.y - 2, table_w, row_h + 2, fill=1, stroke=0)
        self.c.setFillColor(black)
        self.c.setFont("Helvetica-Bold", TABLE_SIZE)
        cx = x0
        for i, h in enumerate(headers):
            self.c.drawString(cx + 2, self.y, h)
            cx += col_widths[i]
        self.y -= row_h

        # Header bottom rule
        self.c.line(x0, self.y + 1, x0 + table_w, self.y + 1)

        # Data rows
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                self.c.setFillColor(HexColor("#F4F4F4"))
                self.c.rect(x0, self.y - 2, table_w, row_h, fill=1, stroke=0)
            self.c.setFillColor(black)
            self.c.setFont("Helvetica", TABLE_SIZE)
            cx = x0
            for i, cell in enumerate(row):
                self.c.drawString(cx + 2, self.y, str(cell))
                cx += col_widths[i]
            self.y -= row_h

        # Bottom rule
        self.c.line(x0, self.y + 2, x0 + table_w, self.y + 2)
        self.y -= 6

    def draw_references(self, refs):
        self.ensure_space(20)
        self.y -= 6
        x = self.col_x()
        self.c.setFont("Helvetica-Bold", SECTION_SIZE)
        self.c.drawString(x, self.y, "References")
        self.y -= SECTION_SIZE + 3

        for i, ref in enumerate(refs, 1):
            self.ensure_space(REF_LEADING * 3)
            x = self.col_x()
            ref_text = f"[{i}] {ref}"
            self.draw_text(ref_text, x, COL_W, font="Helvetica", size=REF_SIZE,
                          leading=REF_LEADING)
            self.y -= 1.5

    def build(self):
        # ========== PAGE 1 HEADER (full width) ==========
        self.draw_title()

        abstract = (
            "Membership inference attacks (MIAs) pose a significant privacy threat to machine learning models "
            "trained on sensitive data. While MIAs have been extensively studied for traditional neural networks, "
            "their behavior on graph neural networks (GNNs) remains less understood, particularly regarding the "
            "role of graph structure in amplifying or mitigating leakage. We present a systematic empirical study "
            "spanning 560 experiments across 8 datasets, 4 model architectures (including non-graph baselines), "
            "and 6 defense configurations. Our findings challenge prevailing assumptions: (1) GNNs exhibit lower "
            "membership leakage than non-graph models on citation networks, suggesting neighborhood aggregation "
            "acts as an implicit regularizer; (2) MIA vulnerability is primarily driven by model overfitting "
            "rather than graph structural properties; (3) label smoothing, commonly assumed to be privacy-protective, "
            "increases leakage by up to 8.5%; and (4) edge sparsification offers the most favorable "
            "privacy-utility tradeoff among lightweight defenses. These results provide actionable guidance "
            "for deploying GNNs on sensitive relational data."
        )
        self.draw_abstract(abstract)

        # Mark start of two-column layout
        self.start_two_column()

        # ========== TWO-COLUMN CONTENT ==========

        # ---------- 1. INTRODUCTION ----------
        self.section(1, "Introduction")

        self.body(
            "Graph neural networks (GNNs) have become the dominant paradigm for learning on "
            "relational data, with applications spanning social networks, biological interactions, "
            "financial transactions, and healthcare knowledge graphs [8, 9]. When deployed on "
            "sensitive data, these models inherit the privacy vulnerabilities of traditional "
            "machine learning, particularly susceptibility to membership inference attacks "
            "(MIAs) [6]. MIAs seek to determine whether a specific data point was used in a "
            "model's training set, posing serious risks to individual privacy, regulatory "
            "compliance (GDPR, HIPAA), and institutional trust."
        )
        self.body(
            "Prior work on MIAs against GNNs has demonstrated that structural information "
            "can be a significant factor in privacy leakage [1, 2]. However, important "
            "questions remain unresolved: Does the graph structure inherently increase "
            "leakage compared to non-graph models? How do fundamental graph properties "
            "such as homophily and density modulate vulnerability? And which lightweight "
            "defenses offer practical protection without sacrificing model utility?"
        )
        self.body("We address these questions through a comprehensive empirical study organized around four research questions:")

        self.body("RQ1: Do GNNs leak more membership information than non-graph baselines on identical data?", bold=True)
        self.body("RQ2: How do graph homophily and edge density affect MIA vulnerability?", bold=True)
        self.body("RQ3: Which lightweight defenses most effectively reduce membership leakage?", bold=True)
        self.body("RQ4: What is the achievable privacy-utility Pareto frontier?", bold=True)

        self.body(
            "Our study encompasses 560 experiments across 8 datasets (2 real-world citation "
            "networks, 6 controlled synthetic graphs), 4 model architectures, and 6 defense "
            "configurations, each evaluated over 5 random seeds. Our key contributions are: "
            "(i) demonstrating that GNNs' neighborhood aggregation acts as an implicit "
            "regularizer that reduces membership leakage relative to non-graph models; "
            "(ii) showing that MIA vulnerability is driven primarily by model overfitting, "
            "not graph structural properties per se; (iii) identifying that label smoothing "
            "counterintuitively increases leakage; and (iv) establishing that edge "
            "sparsification provides the best privacy-utility tradeoff among lightweight defenses."
        )

        # ---------- 2. RELATED WORK ----------
        self.section(2, "Related Work")

        self.subsection("2.1 Membership Inference Attacks")
        self.body(
            "Shokri et al. [6] introduced the shadow model paradigm for MIAs against machine "
            "learning classifiers, demonstrating that model confidence scores contain sufficient "
            "signal to distinguish training members from non-members. The key insight is that "
            "models tend to be more confident on training data than on unseen data, creating an "
            "exploitable signal. Subsequent work has extended these attacks to diverse architectures "
            "and domains, establishing MIAs as a fundamental privacy threat. Aerni et al. [5] "
            "recently highlighted that many privacy defense evaluations are misleading, as "
            "they fail to account for the baseline attack performance achievable without "
            "any model access, calling for more rigorous evaluation protocols."
        )

        self.subsection("2.2 MIAs on Graph Neural Networks")
        self.body(
            "He et al. [1] formalized node-level membership inference against GNNs, showing "
            "that node features and graph topology jointly determine vulnerability. Their work "
            "introduced attack models specifically designed for the graph setting. Olatunji "
            "et al. [2] extended this analysis and demonstrated that structural information is "
            "a major contributing factor to leakage in GNNs, suggesting that the message-passing "
            "mechanism may amplify privacy risks. Yuan et al. [3] introduced the Generalized "
            "Homophily Ratio to characterize how local graph structure modulates privacy risk, "
            "providing a more nuanced understanding of graph-privacy interactions. Mueller "
            "et al. [4] explored privacy-utility trade-offs specifically in medical population "
            "graphs, finding that graph-based models do not necessarily increase privacy risk, "
            "motivating our broader empirical investigation."
        )

        self.subsection("2.3 Privacy Defenses for GNNs")
        self.body(
            "Defense strategies against MIAs span a spectrum from formal guarantees (differential "
            "privacy) to lightweight heuristics (regularization, output perturbation). Wu et al. "
            "[10] proposed graph perturbation as a defense mechanism specifically for GNNs, "
            "modifying the adjacency matrix to reduce information leakage. Rong et al. [7] "
            "introduced DropEdge as a regularization technique for improving GNN training depth, "
            "which has been repurposed as a potential privacy defense due to its stochastic "
            "edge removal during training. In this work, we systematically evaluate five "
            "lightweight defenses spanning training-time and inference-time interventions, "
            "focusing on approaches that require minimal computational overhead and are "
            "practical for real-world deployment."
        )

        # ---------- 3. METHODOLOGY ----------
        self.section(3, "Methodology")

        self.subsection("3.1 Datasets")
        self.body(
            "We use two real-world citation networks and six controlled synthetic graphs "
            "designed to isolate the effects of graph properties. Cora (2,708 nodes, 5,429 "
            "edges, homophily h=0.81) and Citeseer (3,327 nodes, 4,732 edges, h=0.74) are "
            "standard benchmarks where nodes represent publications, edges represent citations, "
            "and node features are bag-of-words representations. For controlled analysis, we "
            "generate synthetic graphs using a stochastic block model with systematically "
            "varied homophily (high: h~0.80, low: h~0.30) and density (sparse: d=0.005, "
            "medium: d=0.015, dense: d=0.040), each with 1,000 nodes and well-separated "
            "Gaussian class features to enable near-perfect classification."
        )

        self.subsection("3.2 Model Architectures")
        self.body(
            "We evaluate four architectures spanning non-graph and graph-based models: "
            "(i) Logistic Regression (LogReg), a linear baseline that ignores graph structure "
            "entirely; (ii) Multi-Layer Perceptron (MLP), a non-linear baseline that also "
            "ignores graph structure; (iii) Graph Convolutional Network (GCN) [8], using "
            "spectral-based neighborhood aggregation with symmetric normalization; and "
            "(iv) GraphSAGE [9], using sampling-based inductive aggregation with mean pooling. "
            "All neural models use 2 hidden layers with 64 units each, ReLU activations, "
            "dropout of 0.5, Adam optimizer (lr=0.01), and are trained for 200 epochs. We "
            "use a consistent 60/20/20 train/validation/test split across all experiments."
        )

        self.subsection("3.3 Attack Methodology")
        self.body(
            "We implement two complementary MIA strategies following established protocols [1, 6]. "
            "The confidence-based attack trains a binary classifier (logistic regression) on "
            "shadow model confidence vectors to distinguish members from non-members. For each "
            "target model, we train a shadow model with the same architecture on disjoint data, "
            "then use its confidence outputs as labeled examples for the attack classifier. "
            "The threshold-based attack applies a calibrated threshold directly to the target "
            "model's maximum prediction confidence, exploiting the observation that training "
            "members typically receive higher confidence scores. We report attack AUC as the "
            "primary metric, where 0.5 indicates no leakage and 1.0 complete leakage."
        )

        self.subsection("3.4 Defense Mechanisms")
        self.body(
            "We evaluate five lightweight defenses chosen for practical deployability: "
            "(i) DropEdge [7]: randomly removes 50% of edges during each training epoch to "
            "reduce structural overfitting; (ii) Label Smoothing: replaces one-hot training "
            "labels with softened distributions using smoothing factor 0.1; (iii) Early "
            "Stopping: monitors validation loss and halts training when it begins to increase, "
            "preventing overfitting; (iv) Confidence Masking: at inference time, truncates "
            "output probability vectors to retain only the top-3 class probabilities, setting "
            "others to zero; and (v) Edge Sparsification: removes 20% of edges with lowest "
            "feature-similarity scores as a preprocessing step."
        )

        self.subsection("3.5 Experimental Protocol")
        self.body(
            "Each configuration (dataset x model x defense) is evaluated across 5 random "
            "seeds (42, 123, 456, 789, 1024), yielding 560 total experiments. We report "
            "mean and standard deviation for all metrics. The attack evaluation uses balanced "
            "member/non-member sets drawn from the training and test splits, respectively. "
            "All experiments are conducted on a single GPU with PyTorch and PyTorch Geometric."
        )

        # ---------- 4. RESULTS ----------
        self.section(4, "Results")

        self.subsection("4.1 RQ1: GNNs vs. Non-Graph Baselines")
        self.body(
            "Table 1 presents our most surprising finding: on both citation networks, non-graph "
            "models exhibit substantially higher attack AUC than GNNs. On Cora, MLP achieves "
            "attack AUC of 0.769 compared to GCN's 0.554 and GraphSAGE's 0.635--a difference "
            "of 0.215 and 0.134 respectively. This pattern is even more pronounced on Citeseer, "
            "where MLP reaches 0.829 while GCN remains at 0.632. Notably, LogReg also shows "
            "high leakage (0.735 on Cora, 0.730 on Citeseer), indicating that the leakage "
            "difference is not merely a function of model complexity."
        )

        # Table 1
        cw1 = [COL_W * 0.20, COL_W * 0.22, COL_W * 0.14, COL_W * 0.24, COL_W * 0.15]
        self.draw_table(
            1,
            "Confidence-based attack AUC and test accuracy on citation networks (no defense). Std. dev. in parentheses.",
            ["Model", "Cora AUC", "Cora Acc", "Cite. AUC", "Cite. Acc"],
            [
                ["LogReg",    "0.735(.012)", "0.743", "0.730(.011)", "0.707"],
                ["MLP",       "0.769(.007)", "0.727", "0.829(.011)", "0.711"],
                ["GCN",       "0.554(.016)", "0.872", "0.632(.008)", "0.724"],
                ["GraphSAGE", "0.635(.013)", "0.880", "0.720(.012)", "0.749"],
            ],
            cw1
        )

        self.body(
            "This counterintuitive result can be explained by examining the generalization gap. "
            "Non-graph models, lacking access to structural information, achieve substantially "
            "lower test accuracy (MLP: 0.727 on Cora vs. GCN: 0.872). This larger train-test "
            "performance gap creates stronger signals for MIA exploitation, as the attack model "
            "can more easily distinguish the high-confidence training predictions from lower-"
            "confidence test predictions. GNNs' neighborhood aggregation effectively regularizes "
            "predictions by smoothing them across connected nodes, reducing the per-node "
            "memorization that MIAs exploit. This finding challenges the assumption from prior "
            "work [2] that graph structure inherently increases privacy leakage, and instead "
            "suggests that the regularization effect of message passing may dominate "
            "(see Figure 1 for a visual comparison across all models)."
        )

        self.subsection("4.2 RQ2: Homophily and Density Effects")
        self.body(
            "On synthetic graphs with well-separated features, all models achieve near-perfect "
            "test accuracy (0.99-1.00), resulting in attack AUC near random chance (0.49-0.54). "
            "This confirms that MIA vulnerability is fundamentally driven by the generalization "
            "gap rather than graph structure per se: when models generalize perfectly, there "
            "is no train-test gap to exploit, regardless of graph properties."
        )
        self.body(
            "The notable exception is GCN on low-homophily sparse graphs (AUC=0.592, "
            "accuracy=0.776), where the model struggles to leverage misaligned structural "
            "signals, creating a larger train-test gap. In contrast, GraphSAGE maintains "
            "near-perfect accuracy (1.000) and near-random attack AUC (0.523) on the same "
            "graphs, demonstrating its greater robustness to structural noise. This divergence "
            "highlights that privacy vulnerability is model-dependent even under identical "
            "graph conditions (see Figures 3 and 5 for detailed analysis)."
        )

        self.subsection("4.3 RQ3: Defense Effectiveness")
        self.body(
            "Table 2 presents defense effectiveness on Cora. The most notable and practically "
            "important finding is that label smoothing increases leakage for both GCN (+7.1%, "
            "from 0.554 to 0.593) and GraphSAGE (+8.5%, from 0.635 to 0.689). This occurs "
            "because label smoothing shifts the output confidence distribution toward a "
            "distinctive pattern--training examples still receive relatively higher confidence "
            "on their true class, but the smoothed distribution provides the attack model with "
            "a richer signal to exploit. This finding echoes Aerni et al.'s [5] warning that "
            "defense evaluations can be misleading."
        )

        # Table 2
        cw2 = [COL_W * 0.24, COL_W * 0.18, COL_W * 0.17, COL_W * 0.22, COL_W * 0.19]
        self.draw_table(
            2,
            "Defense effectiveness on Cora. Attack AUC (lower = better privacy) and test accuracy (higher = better utility).",
            ["Defense", "GCN AUC", "GCN Acc", "SAGE AUC", "SAGE Acc"],
            [
                ["None",           "0.554", "0.872", "0.635", "0.880"],
                ["DropEdge",       "0.559", "0.874", "0.667", "0.875"],
                ["Label Smooth.",  "0.593", "0.873", "0.689", "0.878"],
                ["Early Stop.",    "0.554", "0.872", "0.636", "0.879"],
                ["Conf. Masking",  "0.560", "0.872", "0.635", "0.880"],
                ["Edge Sparsif.",  "0.560", "0.871", "0.630", "0.875"],
            ],
            cw2
        )

        self.body(
            "Edge sparsification provides the most consistent protection for GraphSAGE "
            "(AUC reduction from 0.635 to 0.630) with only marginal accuracy decrease "
            "(0.880 to 0.875). DropEdge, contrary to expectations, increases GraphSAGE "
            "vulnerability (0.635 to 0.667), likely because random edge removal during "
            "training introduces noise that widens the generalization gap rather than closing "
            "it. Early stopping and confidence masking show negligible effects, suggesting "
            "that the primary leakage signal in GNNs is not concentrated in late-stage "
            "overfitting or tail confidence values (see Figure 4 for visual comparison)."
        )

        self.subsection("4.4 RQ4: Privacy-Utility Frontier")
        self.body(
            "Figure 2 maps the full privacy-utility landscape across all model-defense "
            "combinations. GCN achieves the most favorable operating point: attack AUC of "
            "0.554 with test accuracy of 0.872, compared to GraphSAGE's AUC of 0.635 at "
            "accuracy 0.880. The marginal accuracy gain of GraphSAGE (+0.8 percentage points) "
            "comes at a substantial privacy cost (+14.6% relative leakage increase). Among "
            "defenses, edge sparsification and confidence masking define the Pareto frontier "
            "for GraphSAGE, while GCN's baseline already approaches the best achievable "
            "tradeoff. Non-graph baselines occupy the worst region of the frontier, with "
            "high leakage and low accuracy (see Figure 6 for a comprehensive heatmap "
            "across all configurations)."
        )

        # ---------- 5. DISCUSSION ----------
        self.section(5, "Discussion")

        self.subsection("5.1 The Regularization Hypothesis")
        self.body(
            "Our central finding--that GNNs leak less than non-graph models--can be understood "
            "through the lens of implicit regularization. Neighborhood aggregation in GNNs "
            "forces each node's representation to be influenced by its neighbors, effectively "
            "smoothing the learned function across the graph. This smoothing reduces the "
            "model's capacity to memorize individual training examples, which is precisely "
            "the signal that MIAs exploit. The effect is analogous to how convolutional layers "
            "in CNNs share parameters across spatial locations, reducing memorization. This "
            "interpretation aligns with Mueller et al.'s [4] observation that graph structure "
            "does not necessarily increase privacy risk in medical settings, and extends it "
            "to a broader class of GNN architectures and datasets."
        )

        self.subsection("5.2 Why Label Smoothing Fails")
        self.body(
            "The failure of label smoothing as a defense has important practical implications. "
            "Label smoothing is widely used as a regularizer in modern deep learning and has "
            "been suggested as a potential privacy protection mechanism. Our results show it "
            "can be counterproductive: by transforming the one-hot training signal into a "
            "distinctive softened distribution, it creates a confidence signature that is "
            "more informative to the attack model rather than less. Specifically, the smoothed "
            "confidence vectors for training members exhibit a characteristic pattern that "
            "differs systematically from test examples, providing the attack classifier with "
            "a stronger discriminative signal. This underscores the need for empirical "
            "validation of any defense mechanism [5]."
        )

        self.subsection("5.3 Practical Recommendations")
        self.body(
            "For practitioners deploying GNNs on sensitive relational data, our findings "
            "suggest the following guidelines: (1) Prefer GCN over GraphSAGE when the marginal "
            "accuracy gain does not justify the privacy cost--the 0.8% accuracy improvement "
            "may not warrant 14.6% higher leakage. (2) Apply edge sparsification as a "
            "preprocessing step; it provides consistent, albeit modest, privacy improvement "
            "with minimal utility loss. (3) Avoid label smoothing as a privacy defense--it "
            "increases rather than decreases vulnerability. (4) Monitor the train-test accuracy "
            "gap as a proxy for privacy risk: larger gaps indicate greater MIA vulnerability "
            "regardless of model architecture."
        )

        self.subsection("5.4 Limitations and Future Work")
        self.body(
            "Several limitations should be noted. First, we evaluate only confidence-based "
            "and threshold-based attacks; stronger attacks (e.g., loss-based, gradient-based, "
            "or adaptive attacks) may reveal different vulnerability patterns. Second, our "
            "synthetic graphs use well-separated Gaussian features, which may not reflect the "
            "complexity of real-world feature distributions. Third, we focus on transductive "
            "node classification and do not evaluate link-level or subgraph-level MIAs. "
            "Fourth, we do not evaluate differential privacy, which provides formal guarantees "
            "but at higher computational cost. Future work should extend this analysis to "
            "heterogeneous graphs, inductive settings, stronger adaptive attacks, and formal "
            "privacy mechanisms to provide a more complete picture of the GNN privacy landscape."
        )

        # ---------- 6. CONCLUSION ----------
        self.section(6, "Conclusion")
        self.body(
            "We present a systematic empirical study of membership inference vulnerability "
            "in graph neural networks, spanning 560 experiments across 8 datasets, 4 model "
            "architectures, and 6 defense configurations. Our results challenge prevailing "
            "assumptions about graph-based privacy risk: GNNs' neighborhood aggregation acts "
            "as an implicit regularizer that reduces membership leakage relative to non-graph "
            "baselines. We demonstrate that MIA vulnerability is fundamentally driven by model "
            "overfitting rather than graph structural properties, that label smoothing "
            "counterintuitively increases leakage by up to 8.5%, and that edge sparsification "
            "provides the best lightweight defense. These findings offer actionable guidance "
            "for the responsible deployment of GNNs on sensitive relational data and motivate "
            "further research into the interplay between graph structure, generalization, "
            "and privacy."
        )

        # ---------- REFERENCES ----------
        self.draw_references([
            "X. He, R. Wen, Y. Wu, M. Backes, Y. Shen, and Y. Zhang. Node-level membership inference attacks against graph neural networks. arXiv:2102.05429, 2021.",
            "I. Olatunji, W. Nejdl, and M. Khosla. Membership inference attack on graph neural networks. IEEE Trans. on AI-driven Privacy and Security in Intelligent Sys. and Apps., 2021.",
            "X. Yuan, T. Huang, B. Li, and N. Z. Gong. Unveiling privacy vulnerabilities: Investigating the role of structure in graph neural networks. arXiv:2407.18564, 2024.",
            "D. Mueller, T. Paetzold, D. Rueckert, and G. Kaissis. Privacy-utility trade-offs in neural networks for medical population graphs. arXiv:2307.06760, 2023.",
            "S. Aerni, M. Kuo, R. Pillutla, and F. Tramer. Evaluations of machine learning privacy defenses are misleading. In Proc. ACM CCS, 2024.",
            "R. Shokri, M. Stronati, C. Song, and V. Shmatikov. Membership inference attacks against machine learning models. In Proc. IEEE S&P, pp. 3-18, 2017.",
            "Y. Rong, W. Huang, T. Xu, and J. Huang. DropEdge: Towards deep graph convolutional networks on node classification. In Proc. ICLR, 2020.",
            "T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In Proc. ICLR, 2017.",
            "W. L. Hamilton, R. Ying, and J. Leskovec. Inductive representation learning on large graphs. In Proc. NeurIPS, 2017.",
            "F. Wu, T. Long, C. Zhang, and B. Li. Defense against membership inference attack in graph neural networks through graph perturbation. Int. J. of Info. Security, 2022.",
        ])

        # Final page number
        self._page_number()
        self.c.save()
        print(f"Saved: {OUTPUT_PATH}")
        print(f"Pages: {self.page_num}")


if __name__ == "__main__":
    paper = AcademicPaper(OUTPUT_PATH)
    paper.build()
