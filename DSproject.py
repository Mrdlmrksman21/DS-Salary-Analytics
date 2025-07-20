# ------------------- IMPORTS & SETUP -------------------
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QStackedWidget, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QFileDialog, QComboBox, QHeaderView, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ------------------- LOAD DATA -------------------
def load_data(path):
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    return df

# ------------------- DYNAMIC PLOT WINDOW -------------------
class PlotWindow(QWidget):
    def __init__(self, title, fig):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 950, 680)
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(fig)
        # Apply 19px font to everything in the chart window
        canvas.setStyleSheet("font-size: 19px;")
        layout.addWidget(canvas)
        self.setLayout(layout)
        self.show()

# ------------------- MAIN APPLICATION CLASS -------------------
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Science Salary Analytics Dashboard")
        self.setGeometry(100, 60, 1350, 800)
        self.apply_styles()
        self.df = load_data("salaries.csv")
        self.init_label_encoders()
        self._init_ui()

    # --------------- STYLE OVERRIDE ---------------
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #102032; font-size: 1.45em; }
            QLabel, QCheckBox, QComboBox, QPushButton, QTableWidget, QTableWidgetItem {
                font-size: 1.45em;
            }
            QComboBox, QLineEdit {
                min-width: 260px;
                max-width: 420px;
                font-weight: 600;
            }
            QTableWidget {
                background: #1a2938;
                gridline-color: #28f0cf;
                alternate-background-color: #17344a;
                color: #e0ecec;
            }
            QHeaderView::section {
                background-color: #e0ecec;
                color: #000000;
                font-size: 1.3em;
                font-weight: bold;
            }
            QLabel#Header { font-size: 2.7em; font-weight: bold; color: #0ff7dc; }
            QLabel#Footer { color: #1de9b6; font-style: italic; margin-top: 12px; font-size: 1.0em; }
            QListWidget { font-size: 1.55em; font-weight: bold; background: #163138; border-radius: 13px; color: #10ffe8; }
            QListWidget::item:selected { background: #144655; color: #fff; }
            QPushButton { background: #1de9b6; color: #082328; border-radius: 7px; padding: 11px 38px; font-weight: bold; font-size: 1.18em; margin: 11px 0; }
            QPushButton:hover { background: #0bd3a9; color: #fff; }
        """)

    # --------------- LABEL ENCODERS & MODEL ---------------
    def init_label_encoders(self):
        self.encoders = {}
        cols = ["job_title", "experience_level", "company_size", "employment_type",
                "employee_residence", "company_location"]
        for col in cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
        X = self.df[["job_title", "experience_level", "company_size", "employment_type",
                     "remote_ratio", "employee_residence", "company_location"]]
        y = self.df["salary_in_usd"]
        self.predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.predictor.fit(X, y)

    # --------------- MAIN UI LAYOUT INIT ---------------
    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        menu = QListWidget()
        menu.setFixedWidth(250)
        for name in ["View Data", "Graphs & Trends", "Insights", "Salary Predictor", "Exit"]:
            QListWidgetItem(name, menu)
        menu.setCurrentRow(0)
        menu.currentRowChanged.connect(self.display_section)

        self.stack = QStackedWidget()
        self.sections = [
            self.page_table(),
            self.page_charts(),
            self.page_insights(),
            self.page_predictor(),
            self.page_exit()
        ]
        for sec in self.sections:
            self.stack.addWidget(sec)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.addSpacing(17)
        header = QLabel("Data Science\nSalary Analytics")
        header.setObjectName("Header")
        header.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(header)
        sidebar_layout.addSpacing(24)
        sidebar_layout.addWidget(menu)
        sidebar_layout.addStretch()
        footer = QLabel("Powered by Mridul Vaid")
        footer.setAlignment(Qt.AlignHCenter)
        footer.setObjectName("Footer")
        sidebar_layout.addWidget(footer)

        sidebar = QWidget()
        sidebar.setLayout(sidebar_layout)
        sidebar.setFixedWidth(265)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    # --------------- NAVIGATION ---------------
    def display_section(self, idx):
        self.stack.setCurrentIndex(idx)

    # --------------- PAGE: DATA TABLE VIEW ---------------
    def page_table(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        filt_layout = QHBoxLayout()
        self.combo_job = QComboBox()
        self.combo_job.addItem("All")
        jobs = pd.Series(self.encoders["job_title"].classes_).sort_values()
        self.combo_job.addItems(jobs)
        self.combo_job.currentTextChanged.connect(self.refresh_table)
        self.combo_exp = QComboBox()
        self.combo_exp.addItem("All")
        exps = pd.Series(self.encoders["experience_level"].classes_).sort_values()
        self.combo_exp.addItems(exps)
        self.combo_exp.currentTextChanged.connect(self.refresh_table)
        filt_layout.addWidget(QLabel("Job Title:"))
        filt_layout.addWidget(self.combo_job)
        filt_layout.addWidget(QLabel("Experience:"))
        filt_layout.addWidget(self.combo_exp)
        filt_layout.addStretch()
        export_btn = QPushButton("Export this View")
        export_btn.clicked.connect(self.handle_export_data_table)
        filt_layout.addWidget(export_btn)
        layout.addLayout(filt_layout)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        widget.setLayout(layout)
        self.refresh_table()
        return widget

    def refresh_table(self):
        job = self.combo_job.currentText() if hasattr(self, "combo_job") else "All"
        exp = self.combo_exp.currentText() if hasattr(self, "combo_exp") else "All"
        df = self.df.copy()
        if job != "All":
            job_code = list(self.encoders["job_title"].classes_).index(job)
            df = df[df["job_title"] == job_code]
        if exp != "All":
            exp_code = list(self.encoders["experience_level"].classes_).index(exp)
            df = df[df["experience_level"] == exp_code]
        view = df.head(500)
        cols = list(view.columns)
        self.table.setRowCount(len(view))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setStyleSheet("font-size:1.33em;color:#e0ecec;background:#1a2938;")
        header = self.table.horizontalHeader()
        header.setStyleSheet("QHeaderView::section { background-color: #e0ecec; color: #000000; font-size: 1.3em; font-weight: bold; }")
        for i, (_, row) in enumerate(view.iterrows()):
            for j, col in enumerate(cols):
                self.table.setItem(i, j, QTableWidgetItem(str(row[col])))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def handle_export_data_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path: return
        job, exp = self.combo_job.currentText(), self.combo_exp.currentText()
        df = self.df.copy()
        if job != "All":
            job_code = list(self.encoders["job_title"].classes_).index(job)
            df = df[df["job_title"] == job_code]
        if exp != "All":
            exp_code = list(self.encoders["experience_level"].classes_).index(exp)
            df = df[df["experience_level"] == exp_code]
        df.to_csv(path, index=False)

    # --------------- PAGE: GRAPHS & TRENDS ---------------
    def page_charts(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Visible labels with larger font and white color
        chart_type_lbl = QLabel("Chart Type:")
        chart_type_lbl.setStyleSheet("color:#fff;font-size:19px;font-weight:bold;")
        control_layout = QHBoxLayout()
        control_layout.addWidget(chart_type_lbl)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.setStyleSheet("color:#102032; font-size:19px; min-width:200px; height:35px;")
        self.chart_type_combo.addItems([
            "Histogram", "Boxplot", "Line", "Bar", "Pie"
        ])
        control_layout.addWidget(self.chart_type_combo)

        select_cols_lbl = QLabel("Select Columns:")
        select_cols_lbl.setStyleSheet("color:#fff;font-size:19px;font-weight:bold;")
        control_layout.addWidget(select_cols_lbl)
        self.cols_checks_layout = QVBoxLayout()
        cols_widget = QWidget()
        cols_widget.setLayout(self.cols_checks_layout)
        control_layout.addWidget(cols_widget)

        self.visualize_btn = QPushButton("Visualize")
        self.visualize_btn.setStyleSheet("font-size:19px; height:35px;")
        self.visualize_btn.clicked.connect(self.show_selected_chart)
        control_layout.addWidget(self.visualize_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        widget.setLayout(layout)
        self.chart_type_combo.currentIndexChanged.connect(self.update_column_options)
        self.update_column_options(0)
        return widget

    def update_column_options(self, _):
        for i in reversed(range(self.cols_checks_layout.count())):
            widget_to_remove = self.cols_checks_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
        self.check_boxes = []
        chart_type = self.chart_type_combo.currentText()
        if chart_type == "Pie":
            for col in self.df.columns:
                cb = QCheckBox(col)
                cb.setChecked(col == "job_title")
                cb.setStyleSheet("color:#fff;font-size:19px;")
                self.cols_checks_layout.addWidget(cb)
                self.check_boxes.append(cb)
        else:
            wanted_types = {
                "Histogram": ["int64", "float64"],
                "Boxplot": ["int64", "float64", "object"],
                "Line": ["int64", "float64", "object"],
                "Bar": ["int64", "float64", "object"]
            }
            dtypes = wanted_types.get(chart_type, ["int64", "float64", "object"])
            for col in self.df.columns:
                coltype = str(self.df[col].dtype)
                if coltype in dtypes:
                    cb = QCheckBox(col)
                    cb.setChecked(col == "salary_in_usd" or col == "job_title")
                    cb.setStyleSheet("color:#fff;font-size:19px;")
                    self.cols_checks_layout.addWidget(cb)
                    self.check_boxes.append(cb)

    def show_selected_chart(self):
        chart_type = self.chart_type_combo.currentText()
        checked_cols = [cb.text() for cb in self.check_boxes if cb.isChecked()]
        fig, ax = plt.subplots(figsize=(9, 6))
        df = self.df
        if not checked_cols:
            QMessageBox.warning(self, "Field Selection Needed", "Select at least one column to visualize the chart.")
            return
        if chart_type == "Histogram" and len(checked_cols) == 1:
            sns.histplot(df[checked_cols[0]], kde=True, ax=ax, color="#22e6af")
            ax.set_title(f"Histogram of {checked_cols[0]}", color="#0ff7dc")
        elif chart_type == "Boxplot" and len(checked_cols) == 2:
            sns.boxplot(data=df, x=checked_cols[0], y=checked_cols[1], ax=ax, palette="Spectral")
            ax.set_title(f"Boxplot: {checked_cols[1]} by {checked_cols[0]}", color="#0ff7dc")
        elif chart_type == "Line" and len(checked_cols) == 2:
            xs = df.groupby(checked_cols[0])[checked_cols[1]].mean()
            ax.plot(xs.index, xs.values, marker="o", color="#02e6be", linewidth=3.5)
            ax.set_title(f"{checked_cols[1]} over {checked_cols[0]}", color="#0ff7dc")
        elif chart_type == "Bar" and len(checked_cols) == 2:
            gr = df.groupby(checked_cols[0])[checked_cols[1]].mean().sort_values(ascending=False)
            ax.bar(gr.index.astype(str), gr.values, color="#c568ff")
            ax.set_title(f"Bar Chart: Avg. {checked_cols[1]} by {checked_cols[0]}", color="#0ff7dc")
            ax.set_xticklabels(gr.index, rotation=32, ha="right")
        elif chart_type == "Pie":
            for idx, col in enumerate(checked_cols):
                vals = df[col].value_counts()
                ax.pie(vals, labels=vals.index, autopct='%1.1f%%', startangle=140, textprops={'color':"#111"})
                ax.axis('equal')
                ax.set_title(f"Pie of {col}", color="#0ff7dc")
                if idx + 1 < len(checked_cols):
                    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
                    vals2 = df[checked_cols[idx+1]].value_counts()
                    ax2.pie(vals2, labels=vals2.index, autopct='%1.1f%%', startangle=140, textprops={'color':"#111"})
                    ax2.axis('equal')
                    ax2.set_title(f"Pie of {checked_cols[idx+1]}", color="#0ff7dc")
                    PlotWindow(f"{chart_type} - {checked_cols[idx+1]}", fig2)
                break
        else:
            QMessageBox.warning(self, "Select Valid Columns",
                "Histogram/Pie: 1 col | Boxplot/Line/Bar: 2 cols (group, value)")
            return
        fig.tight_layout()
        self.plot_win = PlotWindow(f"{chart_type} - {checked_cols}", fig)

    # --------------- PAGE: INSIGHTS (PROFESSIONAL VISUALS) ---------------
    def page_insights(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addSpacing(10)
        df = self.df
        # Card 1: Top Paying Roles
        card1 = QWidget()
        card1.setStyleSheet(
            "background:#172230; border-radius:16px; padding:22px; margin-bottom:16px;"
            "box-shadow: 0 0 13px #6be97a;"
        )
        lbl1 = QLabel("💸 <b>Top 5 Highest Paying Roles</b>")
        lbl1.setStyleSheet("font-size: 19px; font-weight:bold; color:#fff;")
        labels_jobs = self.encoders["job_title"].classes_
        job_mean = df.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False)[:5]
        job_mean_names = [labels_jobs[i] for i in job_mean.index]
        valstr = "<br>".join([f"<b>{j}</b>: <span style='font-size:19px;color:#00ffc0;'>${int(s):,}</span>" for j, s in zip(job_mean_names, job_mean)])
        msg1 = QLabel(valstr)
        msg1.setTextFormat(Qt.RichText)
        msg1.setStyleSheet("font-size: 19px;")
        card1_lay = QVBoxLayout(); card1_lay.addWidget(lbl1); card1_lay.addWidget(msg1)
        card1.setLayout(card1_lay)
        # Card 2: Top Salary Countries
        card2 = QWidget()
        card2.setStyleSheet(
            "background:#23263a; border-radius:16px; padding:22px; margin-bottom:14px;"
            "box-shadow: 0 0 8px #cf9cff;"
        )
        lbl2 = QLabel("🌎 <b>Top 5 Countries by Avg. Salary</b>")
        lbl2.setStyleSheet("font-size:19px;font-weight:bold; color:#fff;")
        labels_countries = self.encoders["employee_residence"].classes_
        country_mean = df.groupby("employee_residence")["salary_in_usd"].mean().sort_values(ascending=False)[:5]
        country_mean_names = [labels_countries[i] for i in country_mean.index]
        valstr2 = "<br>".join([f"<b>{c}</b>: <span style='font-size:19px;color:#e184ff;'>${int(s):,}</span>" for c, s in zip(country_mean_names, country_mean)])
        msg2 = QLabel(valstr2)
        msg2.setTextFormat(Qt.RichText)
        msg2.setStyleSheet("font-size: 19px;")
        card2_lay = QVBoxLayout(); card2_lay.addWidget(lbl2); card2_lay.addWidget(msg2)
        card2.setLayout(card2_lay)
        # Card 3: Distinct Counts
        card3 = QWidget()
        card3.setStyleSheet(
            "background:#1c2439; border-radius:13px; padding:16px; margin-bottom:7px;"
            "box-shadow: 0 0 8px #1de9b6;"
        )
        n_jobs = df["job_title"].nunique()
        n_countries = df["employee_residence"].nunique()
        jobs_lbl = QLabel(f"📊 <b>Distinct Roles:</b> <span style=\"font-size:19px;color:#6be97a;\">{n_jobs}</span>")
        ctry_lbl = QLabel(f"🗺️ <b>Distinct Countries:</b> <span style=\"font-size:19px;color:#24fbff;\">{n_countries}</span>")
        jobs_lbl.setStyleSheet("font-size:19px;"); ctry_lbl.setStyleSheet("font-size:19px;")
        card3_lay = QVBoxLayout(); card3_lay.addWidget(jobs_lbl); card3_lay.addWidget(ctry_lbl)
        card3.setLayout(card3_lay)
        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addWidget(card3)
        layout.addSpacing(25)
        widget.setLayout(layout)
        return widget

    # --------------- PAGE: ENGAGING SALARY PREDICTOR ---------------
    def page_predictor(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        title_lbl = QLabel("Live Salary Prediction Terminal")
        title_lbl.setStyleSheet(
            "color: #00ffc0; font-weight: bold; font-size: 19px; font-family: 'Consolas', monospace; margin-bottom:22px;"
        )
        layout.addWidget(title_lbl)
        options = {}
        for col in ["job_title", "experience_level", "company_size", "employment_type", "employee_residence", "company_location"]:
            field_layout = QHBoxLayout()
            label = QLabel(col.replace("_", " ").title() + ":")
            label.setFixedWidth(188)
            label.setStyleSheet("color:#fff;font-size:19px;font-family: 'Consolas', monospace;")
            combo = QComboBox()
            combo.addItems(self.encoders[col].classes_)
            combo.setStyleSheet(
                "background: #fff; color: #0d2a3b; font-size: 19px; font-family: 'Consolas', monospace;"
                "border: 2px solid #09e88f; border-radius:11px; padding: 14px 12px;"
            )
            options[col] = combo
            field_layout.addWidget(label)
            field_layout.addWidget(combo)
            layout.addLayout(field_layout)
        # Remote ratio
        field_layout = QHBoxLayout()
        label = QLabel("Remote Ratio:")
        label.setFixedWidth(188)
        label.setStyleSheet("color:#fff;font-size:19px;font-family: 'Consolas', monospace;")
        remote_combo = QComboBox()
        remote_combo.addItems(["0", "50", "100"])
        remote_combo.setStyleSheet(
            "background: #fff; color: #0d2a3b; font-size: 19px; font-family: 'Consolas', monospace;"
            "border: 2px solid #e8df09; border-radius:11px; padding: 14px 12px;"
        )
        options["remote_ratio"] = remote_combo
        field_layout.addWidget(label)
        field_layout.addWidget(remote_combo)
        layout.addLayout(field_layout)
        # Prediction
        pred_btn = QPushButton("PREDICT ✦")
        pred_btn.setStyleSheet(
            "background-color:#0cff31; color:#112233; border-radius:16px;"
            "font-weight:bold; font-size:19px; font-family:'Consolas',monospace; box-shadow: 0 0 10px #0cff60;"
            "padding: 16px 48px; margin-top:25px; margin-bottom:3px;"
        )
        pred_label = QLabel("")
        pred_label.setStyleSheet(
            "font-weight: bold; color: #09e88f; font-size: 19px; background: #09202f; "
            "border-radius:14px; font-family: 'Consolas', monospace; padding: 19px 0 19px 30px; margin-top:9px;"
        )
        indicator = QLabel("")
        indicator.setStyleSheet(
            "color:#f7ff94; font-size:19px; background: #204440;"
            "font-family: 'Consolas', monospace; border-radius:7px;"
        )
        def do_predict():
            vals = []
            for col in [
                "job_title", "experience_level", "company_size", "employment_type",
                "remote_ratio", "employee_residence", "company_location"
            ]:
                if col == "remote_ratio":
                    vals.append(int(options[col].currentText()))
                else:
                    val = options[col].currentText()
                    code = list(self.encoders[col].classes_).index(val)
                    vals.append(code)
            arr = pd.DataFrame([vals], columns=[
                "job_title", "experience_level", "company_size", "employment_type",
                "remote_ratio", "employee_residence", "company_location"
            ])
            pred = int(self.predictor.predict(arr)[0])
            import random
            step = random.randint(-1100, 1100)
            color = "#21fa81" if step > 0 else "#ff3e3e"
            indicator.setText(
                f"<span style='color:{color};'>{'▲' if step>0 else '▼'} ${abs(step):,} | Δ</span>")
            pred_label.setText(f"Estimated Salary: ${pred:,}")
        pred_btn.clicked.connect(do_predict)
        layout.addWidget(pred_btn)
        layout.addWidget(pred_label)
        layout.addWidget(indicator)
        widget.setLayout(layout)
        return widget

    # --------------- PAGE: EXIT ---------------
    def page_exit(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        label = QLabel("Thank you for using the Salary Analytics Dashboard.")
        label.setFont(QFont('Arial', 22, QFont.Bold))
        label.setStyleSheet("color: #1de9b6; padding: 35px 0; font-size:19px;")
        layout.addWidget(label)
        exit_btn = QPushButton("Close Application")
        exit_btn.setStyleSheet(
            "background: #e53935; color: white; font-size: 19px; border-radius: 10px; "
            "padding: 14px 50px; margin-top: 16px;"
        )
        exit_btn.clicked.connect(QApplication.quit)
        layout.addWidget(exit_btn)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

# ------------------- MAIN EXECUTION -------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())

