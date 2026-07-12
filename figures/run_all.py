"""
Run all figure-generation scripts for the MDA-AENMF-AD manuscript.
Usage:  python run_all.py
Output: PNG files written to ./figures/
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "fig2_performance.py",
    "fig3_ablation.py",
    "fig4_sensitivity.py",
    "fig5_ms_predictions.py",
    "fig6_crossdisease.py",
    "fig7_disease_predictions.py",
    "fig8_pathway_themes.py",
    "fig9_litval_config.py",
]

if __name__ == "__main__":
    for s in SCRIPTS:
        print(f"\n=== Running {s} ===")
        subprocess.run([sys.executable, os.path.join(HERE, s)], check=True, cwd=HERE)
    print("\nAll figures written to ./figures/")
