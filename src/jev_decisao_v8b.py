"""Segunda consulta ao Jev (2026-10-19), com a evidência nova: folha cega de 222 itens do 4B v7b e a
validação fora da amostra do juiz Jev. Reutiliza o cliente e a composição de src/jev_decisao_v8.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_decisao_v8 as j  # noqa: E402

j.OUT = Path(__file__).resolve().parent.parent / "outputs" / "melhoria_v8" / "jev_decisao_2_resultado.json"
j.OPCOES["B"] = ("Rejection-sampling fine-tuning and preference training: generate many samples with the 4B, keep the ones "
                 "that the arithmetic verifier and the validated online judge both accept, and train on accepted samples and on accepted-vs-rejected pairs.")
j.OPCOES["F"] = ("Collect a second batch of about 220 teacher ratings of the current 4B model to narrow the confidence interval further.")
j.STATE["options"] = j.OPCOES
j.STATE["measurement_today"] = {
    "human_blind_sample": "222 items of the current 4B fine-tuned model rated blind by the teacher, plus 30 earlier ones; the 95% interval is about plus or minus 6 percentage points.",
    "online_judge": "A judge service reached over the internet was validated out of sample against the teacher on those 222 items with a threshold fixed beforehand: agreement kappa 0.80 (interval 0.70 to 0.88), 90% of rejected items were truly invalid, 81% of invalid items were caught, 3.9% of valid items were lost. Items it accepted were teacher-valid 92% of the time against 69% without it. It costs about US$0.00005 per item. It cannot run on the phone because the app is fully offline.",
    "mechanical_gate": "Two trainings with identical configuration gave opposite verdicts, so the gate decides inside training noise.",
}
j.STATE["measured_results"] = {
    "teacher_core_validity_current_4b": "68.9% of 222 items (interval 62.5 to 74.6). By school year: 9th grade 54%, 5th grade 70%, 2nd grade 82%. By requested difficulty: Hard 54%, Moderate 74%, Easy 78%.",
    "verifier_vs_teacher": "Items the arithmetic verifier marked ok were teacher-valid 83% of the time; items marked not verifiable 58%.",
    "earlier": "The 4B fine-tuned model equals the 4B zero-shot on teacher validity; the training data itself is 97.9% valid.",
}
j.STATE["teacher_findings_4b_finetuned"] = [
    "Ill-posed problem: contradictory, impossible, underdetermined or invented data in the statement. 24 of 69 failures (35%).",
    "Wrong calculation or wrong concept in the model's own solution or marked answer. 19 of 69 (28%).",
    "More than one option is correct, for example equivalent fractions. 14 of 69 (20%).",
    "The solution reaches the correct value, but the options do not contain it or the marked letter is wrong. 12 of 69 (17%), which is about 5 points of the 31 points of total failure.",
]
j.STATE["option_A_ceiling"] = "Option A only addresses the last finding, so its best case is about 5 percentage points of core validity."
if __name__ == "__main__":
    j.main()
