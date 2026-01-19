# 🔍 AI Safety & Calibration

Empirical evaluation and mechanistic interpretability analysis of transformer models, with focus on reliability, calibration, and understanding internal decision mechanisms using GPT-2.

## 🎯 Overview

This project explores **trustworthiness in language models** through:

- **Calibration Analysis**: How confident should models be in their predictions?
- **Direct Logit Attribution (DLA)**: Understanding which tokens influence predictions
- **Selective Prediction**: Teaching models when to abstain from uncertain predictions
- **Mechanistic Interpretability**: Visualizing internal decision processes

## 📊 Key Research Questions

1. **Calibration**: Do model confidence scores match actual accuracy?
2. **Interpretability**: Which input tokens most influence predictions?
3. **Uncertainty**: Can we identify when models are unreliable?
4. **Safety**: How do we prevent overconfident incorrect predictions?

## 🛠️ Tech Stack

- **Model**: GPT-2 (124M parameters)
- **Framework**: PyTorch, TransformerLens
- **Analysis**: Direct Logit Attribution, attention visualization
- **Calibration**: Temperature scaling, Platt scaling
- **Evaluation**: Expected Calibration Error (ECE), Brier Score

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/farunawebservices/gpt2-safety-calibration.git
cd gpt2-safety-calibration

# Install dependencies
pip install -r requirements.txt

# Download GPT-2 model
python download_model.py

# Run analysis
python run_calibration_analysis.py

🔍 Usage
Calibration Analysis
from calibration import CalibrationAnalyzer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize analyzer
analyzer = CalibrationAnalyzer(model, tokenizer)

# Evaluate calibration
results = analyzer.evaluate_calibration(
    dataset="wikitext",
    num_samples=1000
)

print(f"Expected Calibration Error: {results['ece']:.4f}")
print(f"Brier Score: {results['brier']:.4f}")

Direct Logit Attribution

from interpretability import DLAAnalyzer

# Analyze token influence
dla = DLAAnalyzer(model)
text = "The capital of France is"

attributions = dla.compute_attributions(text)
dla.visualize_attributions(attributions)

# Output shows which tokens contribute to predicting "Paris"

Selective Prediction
from selective_prediction import SelectivePredictor

# Model abstains on uncertain predictions
predictor = SelectivePredictor(model, threshold=0.7)

result = predictor.predict_with_abstention(
    text="The capital of France is",
    confidence_threshold=0.7
)

if result['abstain']:
    print("Model is uncertain - abstaining from prediction")
else:
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")

📈 Experimental Results
Calibration Improvements
| Method              | ECE (Before) | ECE (After) | Improvement |
| ------------------- | ------------ | ----------- | ----------- |
| Temperature Scaling | 0.142        | 0.089       | 37.3% ↓     |
| Platt Scaling       | 0.142        | 0.095       | 33.1% ↓     |
| Isotonic Regression | 0.142        | 0.091       | 35.9% ↓     |

Lower ECE = better calibration

DLA Insights
Example: "The capital of France is ____"

Token attributions for predicting "Paris":

"France" → +2.34 (strongest positive influence)

"capital" → +1.87

"The" → +0.42

"is" → +0.18

Selective Prediction Results
| Confidence Threshold | Coverage | Accuracy |
| -------------------- | -------- | -------- |
| 0.5 (no abstention)  | 100%     | 76.3%    |
| 0.7                  | 87%      | 84.2%    |
| 0.9                  | 64%      | 92.1%    |

Higher threshold = fewer predictions but higher accuracy

⚠️ Limitations
Model Size: Tested on GPT-2 (124M); results may not generalize to larger models (GPT-3, GPT-4)

Dataset: Evaluated on WikiText; domain-specific calibration may differ

Calibration Methods: Post-hoc calibration; doesn't address fundamental model uncertainty

Interpretability: DLA shows correlation, not causation

Computational: Full attribution analysis requires significant memory

Production: Selective prediction reduces coverage; tradeoff with utility

🔬 Technical Deep Dive
Expected Calibration Error (ECE)
Measures gap between predicted confidence and actual accuracy:
def compute_ece(confidences, accuracies, num_bins=10):
    """
    ECE = Σ |confidence - accuracy| × (bin_count / total)
    """
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0
    
    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = accuracies[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += abs(bin_confidence - bin_accuracy) * bin_weight
    
    return ece

Temperature Scaling
Simple post-hoc calibration method:
# Before: p(y|x) = softmax(logits)
# After:  p(y|x) = softmax(logits / T)

# Find optimal temperature on validation set
def find_optimal_temperature(logits, labels):
    temperatures = np.linspace(0.1, 5.0, 100)
    best_ece = float('inf')
    best_temp = 1.0
    
    for T in temperatures:
        calibrated_probs = softmax(logits / T)
        ece = compute_ece(calibrated_probs, labels)
        if ece < best_ece:
            best_ece = ece
            best_temp = T
    
    return best_temp

🔮 Future Work
 Test on larger models (GPT-3, GPT-4, LLaMA)

 Implement conformal prediction for uncertainty quantification

 Compare to ensemble methods (multiple model voting)

 Add domain-specific calibration (code, math, medical)

 Explore in-context calibration (without fine-tuning)

 Build real-time calibration monitoring dashboard

📚 References
Guo et al. (2017) - On Calibration of Modern Neural Networks

Nanda et al. (2023) - TransformerLens: Mechanistic Interpretability

Geifman & El-Yaniv (2017) - Selective Prediction for Neural Networks

Kadavath et al. (2022) - Language Models (Mostly) Know What They Know

📄 License
MIT License - See LICENSE for details

🙏 Acknowledgments
TransformerLens team (Neel Nanda)

Anthropic interpretability research

OpenAI GPT-2 release

📧 Contact
Faruna Godwin Abuh
Applied AI Safety Engineer
📧 farunagodwin01@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/faruna-godwin-abuh-07a22213b/

