import json
import subprocess
import numpy as np
from datetime import datetime
import mlflow
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import os
import sys
import time
import logging

# Add the config_scripts directory to the path for logging config
sys.path.append('/workspace')
sys.path.append('.')

try:
    from logging_config import setup_logging, get_log_config
except ImportError:
    # Fallback logging setup if logging_config is not available
    def setup_logging(name="mlflow_evaluation", log_level="INFO", **kwargs):
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/output/evaluation.log' if os.path.exists('/output') else 'evaluation.log')
            ]
        )
        return logging.getLogger(name)


    def get_log_config():
        return {'log_level': 'INFO'}

# Initialize logging
log_config = get_log_config()
logger = setup_logging(
    name="evaluation_pipeline",
    log_level=log_config.get('log_level', 'INFO'),
    log_dir=log_config.get('log_dir', '/output/logs'),
    console_output=True
)

# Download NLTK data with error handling
try:
    logger.info("Downloading NLTK punkt tokenizer data...")
    nltk.download('punkt_tab', quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

# Environment variables for Kubernetes deployment
QUERIER_API_ENDPOINT = os.environ.get("QUERIER_API_ENDPOINT", "http://localhost:8080/query")
PARAMS_QUERIER_ENDPOINT = os.environ.get("PARAMS_QUERIER_ENDPOINT", "http://localhost:8080/params")
API_KEY = os.environ.get("API_KEY", "default-key")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")

logger.info(f"Configuration loaded:")
logger.info(f"  QUERIER_API_ENDPOINT: {QUERIER_API_ENDPOINT}")
logger.info(f"  PARAMS_QUERIER_ENDPOINT: {PARAMS_QUERIER_ENDPOINT}")
logger.info(f"  MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
logger.info(f"  API_KEY: {'*' * len(API_KEY) if API_KEY != 'default-key' else 'default-key'}")

QUESTIONS_FILE = "questions.txt"
BLEU_THRESHOLD = 0.5
ROUGE_L_THRESHOLD = 0.6
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

logger.info(f"Evaluation thresholds - BLEU: {BLEU_THRESHOLD}, ROUGE-L: {ROUGE_L_THRESHOLD}")


def calculate_bleu(gen, ref):
    """Calculate BLEU score between generated and reference text."""
    try:
        logger.debug(f"Calculating BLEU score")
        score = sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(gen),
                              smoothing_function=SmoothingFunction().method1)
        logger.debug(f"BLEU score calculated: {score:.4f}")
        return score
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        return 0.0


def calculate_rouge_l(gen, ref):
    """Calculate ROUGE-L score between generated and reference text."""
    try:
        logger.debug(f"Calculating ROUGE-L score")
        score = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, gen)["rougeL"].fmeasure
        logger.debug(f"ROUGE-L score calculated: {score:.4f}")
        return score
    except Exception as e:
        logger.error(f"Error calculating ROUGE-L: {e}")
        return 0.0


def parse_qna_file(path):
    """Parse Q&A file into list of (question, answer) tuples."""
    logger.info(f"Parsing Q&A file: {path}")
    qna = []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read Q&A file, {len(content)} characters")
    except FileNotFoundError:
        logger.error(f"Questions file '{path}' not found")
        return []
    except Exception as e:
        logger.error(f"Error reading questions file: {e}")
        return []

    parts = content.split("\nQuestion: ")
    logger.debug(f"Split content into {len(parts)} parts")

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        if not part.startswith("Question: "):
            part = "Question: " + part

        lines = part.splitlines()
        question_lines = []
        answer_lines = []
        is_answer = False

        for line in lines:
            if line.startswith("Question: "):
                question_lines.append(line[len("Question: "):].strip())
            elif line.startswith("Answer:"):
                is_answer = True
                answer_start = line[len("Answer:"):].strip()
                if answer_start:
                    answer_lines.append(answer_start)
            else:
                if is_answer:
                    answer_lines.append(line)
                else:
                    question_lines.append(line)

        question = " ".join(question_lines).strip()
        answer = "\n".join(answer_lines).strip()

        if question and answer:
            qna.append((question, answer))
            logger.debug(f"Parsed Q&A pair {len(qna)}: {question[:50]}...")
        else:
            logger.warning(
                f"Skipping incomplete Q&A pair {i}: question='{question[:30]}...', answer='{answer[:30]}...'")

    logger.info(f"Successfully parsed {len(qna)} Q&A pairs")
    return qna


from urllib.parse import quote


def call_querier(question):
    """Call the querier API with retry logic."""
    logger.debug(f"Calling querier API for question: {question[:100]}...")

    for attempt in range(MAX_RETRIES):
        try:
            encoded_question = quote(question)
            url = f"{QUERIER_API_ENDPOINT}?queryText={encoded_question}"

            logger.debug(f"API call attempt {attempt + 1}/{MAX_RETRIES} to: {QUERIER_API_ENDPOINT}")

            result = subprocess.run(
                ["curl", "-s", "-X", "GET", url,
                 "-H", f"apikey: {API_KEY}",
                 "--connect-timeout", "30",
                 "--max-time", "60"],
                capture_output=True, text=True, timeout=70
            )

            if result.returncode != 0:
                logger.warning(f"Curl failed with return code {result.returncode}: {result.stderr}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"All API call attempts failed for question")
                return ""

            if not result.stdout.strip():
                logger.warning(f"Empty response from API (attempt {attempt + 1})")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Received empty response after all retries")
                return ""

            response_json = json.loads(result.stdout)
            answer = response_json.get("answer", "")
            logger.debug(f"API call successful, received answer: {answer[:100]}...")
            return answer

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
            logger.debug(f"Response content: {result.stdout[:200]}...")
        except subprocess.TimeoutExpired:
            logger.error(f"API call timeout (attempt {attempt + 1})")
        except Exception as e:
            logger.error(f"Unexpected error in API call (attempt {attempt + 1}): {e}")

        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY * (attempt + 1)
            logger.info(f"Retrying API call in {wait_time} seconds...")
            time.sleep(wait_time)

    logger.error(f"Failed to get response after {MAX_RETRIES} attempts")
    return ""


def get_parameters():
    """Get configurable parameters from the API."""
    logger.info("Retrieving configurable parameters from API...")

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Parameters API call attempt {attempt + 1}/{MAX_RETRIES}")

            result = subprocess.run(
                ["curl", "-s", "-X", "GET", PARAMS_QUERIER_ENDPOINT,
                 "-H", f"apikey: {API_KEY}",
                 "--connect-timeout", "30",
                 "--max-time", "60"],
                capture_output=True, text=True, timeout=70
            )

            if result.returncode != 0:
                logger.warning(f"Parameters curl failed: {result.stderr}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                logger.error("All parameter retrieval attempts failed")
                return {}

            if not result.stdout.strip():
                logger.warning(f"Empty response from parameters API (attempt {attempt + 1})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                logger.error("No parameters received after all retries")
                return {}

            response_json = json.loads(result.stdout)
            logger.info(f"Successfully retrieved {len(response_json)} parameters")
            logger.debug(f"Parameters: {response_json}")
            return response_json

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for parameters (attempt {attempt + 1}): {e}")
        except subprocess.TimeoutExpired:
            logger.error(f"Parameters API call timeout (attempt {attempt + 1})")
        except Exception as e:
            logger.error(f"Error getting parameters (attempt {attempt + 1}): {e}")

        if attempt < MAX_RETRIES - 1:
            logger.info(f"Retrying parameter retrieval in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

    logger.warning("Using empty parameters after all retry attempts failed")
    return {}


def wait_for_mlflow():
    """Wait for MLflow to be available."""
    logger.info("Checking MLflow availability...")

    for attempt in range(30):  # Wait up to 5 minutes
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.search_experiments()
            logger.info("MLflow is available and responding!")
            return True
        except Exception as e:
            if attempt == 0:
                logger.info(f"MLflow not ready yet, waiting... (will try for 5 minutes)")
            elif attempt % 6 == 0:  # Log every minute
                logger.info(f"Still waiting for MLflow... (attempt {attempt + 1}/30)")
            logger.debug(f"MLflow connection attempt {attempt + 1} failed: {e}")
            time.sleep(10)

    logger.error("Could not connect to MLflow after 5 minutes")
    return False


def main():
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("STARTING MLFLOW EVALUATION PIPELINE")
    logger.info("=" * 60)

    start_time = datetime.now()
    logger.info(f"Evaluation started at: {start_time}")

    # Wait for MLflow to be available
    if not wait_for_mlflow():
        logger.error("Proceeding without MLflow connection (this may cause issues)...")

    # Set MLflow tracking URI
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.error(f"Error setting MLflow tracking URI: {e}")
        return

    # Set experiment
    try:
        mlflow.set_experiment("agent_pipeline")
        logger.info("MLflow experiment set to: agent_pipeline")
    except Exception as e:
        logger.error(f"Error setting MLflow experiment: {e}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"Run_{timestamp}"
    logger.info(f"Starting MLflow run: {run_name}")

    try:
        with mlflow.start_run(run_name=run_name) as main_run:
            logger.info(f"MLflow run started with ID: {main_run.info.run_id}")

            # Log configurable parameters
            logger.info("Logging configurable parameters...")
            with mlflow.start_run(run_name="configurable_parameters", nested=True) as param_run:
                params = get_parameters()
                logger.info(f"Retrieved {len(params)} configurable parameters")

                for key, value in params.items():
                    try:
                        mlflow.log_param(key, value)
                        logger.debug(f"Logged parameter: {key} = {value}")
                    except Exception as e:
                        logger.error(f"Error logging parameter {key}: {e}")

            # Parse Q&A pairs
            qna_pairs = parse_qna_file(QUESTIONS_FILE)
            if not qna_pairs:
                logger.error("No Q&A pairs found. Exiting evaluation.")
                return

            logger.info(f"Starting evaluation of {len(qna_pairs)} Q&A pairs")

            all_bleu, all_rouge = [], []
            flagged_count = 0
            successful_evaluations = 0

            # Output directory for Kubernetes
            output_dir = "/output" if os.path.exists("/output") else "."
            output_file = os.path.join(output_dir, "generated_answers.txt")
            logger.info(f"Results will be saved to: {output_file}")

            with open(output_file, "w", encoding='utf-8') as out:
                out.write(f"MLflow Evaluation Results - {timestamp}\n")
                out.write(f"Run ID: {main_run.info.run_id}\n")
                out.write(f"Total Questions: {len(qna_pairs)}\n")
                out.write("=" * 80 + "\n\n")

                for i, (question, expected) in enumerate(qna_pairs):
                    logger.info(f"Processing question {i + 1}/{len(qna_pairs)}")
                    logger.debug(f"Question: {question[:100]}...")

                    # Call API
                    generated = call_querier(question)

                    if not generated:
                        logger.warning(f"Empty response for question {i + 1}, using placeholder")
                        generated = "[No response received]"

                    # Calculate metrics
                    bleu = calculate_bleu(generated, expected)
                    rouge = calculate_rouge_l(generated, expected)

                    all_bleu.append(bleu)
                    all_rouge.append(rouge)

                    flagged = (bleu < BLEU_THRESHOLD) or (rouge < ROUGE_L_THRESHOLD)
                    if flagged:
                        flagged_count += 1
                        logger.warning(f"Question {i + 1} FLAGGED - BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}")
                    else:
                        logger.debug(f"Question {i + 1} OK - BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}")

                    successful_evaluations += 1

                    # Write detailed results
                    out.write(f"Question {i + 1}:\n")
                    out.write(f"Q: {question}\n")
                    out.write(f"Expected: {expected}\n")
                    out.write(f"Generated: {generated}\n")
                    out.write(f"BLEU: {bleu:.4f} | ROUGE-L: {rouge:.4f}\n")
                    out.write(f"Status: {'FLAGGED' if flagged else 'OK'}\n")
                    out.write("-" * 80 + "\n\n")

            # Calculate final metrics
            avg_bleu = np.mean(all_bleu) if all_bleu else 0
            avg_rouge = np.mean(all_rouge) if all_rouge else 0
            flagged_pct = (flagged_count / len(qna_pairs)) * 100 if qna_pairs else 0

            logger.info("=" * 40)
            logger.info("EVALUATION SUMMARY")
            logger.info("=" * 40)
            logger.info(f"Total Questions: {len(qna_pairs)}")
            logger.info(f"Successful Evaluations: {successful_evaluations}")
            logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
            logger.info(f"Average ROUGE-L Score: {avg_rouge:.4f}")
            logger.info(f"Flagged Questions: {flagged_count} ({flagged_pct:.2f}%)")

            # Log metrics to MLflow
            metrics = {
                "Avg_BLEU": avg_bleu,
                "Avg_ROUGE_L": avg_rouge,
                "Flagged_Percentage": flagged_pct,
                "Total_Questions": len(qna_pairs),
                "Flagged_Count": flagged_count,
                "Successful_Evaluations": successful_evaluations
            }

            try:
                mlflow.log_metrics(metrics)
                logger.info("Metrics successfully logged to MLflow")
            except Exception as e:
                logger.error(f"Error logging metrics to MLflow: {e}")

            # Append summary to file
            with open(output_file, "a", encoding='utf-8') as out:
                out.write("\n" + "=" * 80 + "\n")
                out.write("FINAL SUMMARY\n")
                out.write("=" * 80 + "\n")
                out.write(f"Evaluation completed at: {datetime.now()}\n")
                out.write(f"Total Questions: {len(qna_pairs)}\n")
                out.write(f"Successful Evaluations: {successful_evaluations}\n")
                out.write(f"Average BLEU: {avg_bleu:.4f}\n")
                out.write(f"Average ROUGE-L: {avg_rouge:.4f}\n")
                out.write(f"Flagged Count: {flagged_count}\n")
                out.write(f"Flagged Percentage: {flagged_pct:.2f}%\n")

                # Performance metrics
                end_time = datetime.now()
                duration = end_time - start_time
                out.write(f"Total Execution Time: {duration}\n")
                if len(qna_pairs) > 0:
                    avg_time_per_question = duration.total_seconds() / len(qna_pairs)
                    out.write(f"Average Time per Question: {avg_time_per_question:.2f} seconds\n")

            # Log artifacts
            try:
                if os.path.exists(QUESTIONS_FILE):
                    mlflow.log_artifact(QUESTIONS_FILE)
                    logger.info("Questions file logged as artifact")
                mlflow.log_artifact(output_file)
                logger.info("Results file logged as artifact")
            except Exception as e:
                logger.error(f"Error logging artifacts: {e}")

            logger.info(f"Evaluation complete. Results saved to {output_file}")

            # Generate plot
            logger.info("Generating visualization...")
            generate_plot()

            # Final timing
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Total evaluation time: {duration}")
            logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.critical(f"Critical error in main evaluation: {e}")
        logger.exception("Full exception traceback:")
        raise


def safe_max(lst, default=0):
    """Safely get max value from list."""
    return max(lst) if lst else default


def generate_plot():
    """Generate interactive plot of evaluation metrics."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from html import escape

        logger.info("Starting plot generation...")

        with mlflow.start_run(run_name="plot", nested=True) as plot_run:
            logger.debug(f"Plot run started with ID: {plot_run.info.run_id}")

            try:
                runs_df = mlflow.search_runs(experiment_names=["agent_pipeline"],
                                             filter_string="attributes.status = 'FINISHED'")
                logger.info(f"Found {len(runs_df)} finished runs for plotting")
            except Exception as e:
                logger.error(f"Error searching MLflow runs: {e}")
                return

            main_runs = runs_df[runs_df["tags.mlflow.runName"].str.startswith("Run_")]
            logger.info(f"Filtered to {len(main_runs)} main runs for plotting")

            if main_runs.empty:
                logger.warning("No runs found for plotting")
                return

            bleu_scores, rouge_scores = [], []
            run_ids, x_vals, tick_vals, tick_texts, tooltips = [], [], [], [], []

            tracking_uri = mlflow.get_tracking_uri()
            base_url = tracking_uri.rstrip("/")
            logger.debug(f"Using MLflow base URL: {base_url}")

            for idx, (_, row) in enumerate(main_runs.iterrows()):
                run_id = row["run_id"]
                label = f"Run_{idx + 1}"
                run_link = f"{base_url}/#/experiments/{row['experiment_id']}/runs/{run_id}"

                tick_vals.append(label)
                tick_texts.append(
                    f'<a href="{run_link}" target="_blank" style="color:blue;text-decoration:underline;">{label}</a>')
                x_vals.append(label)

                bleu = row.get("metrics.Avg_BLEU", 0)
                rouge = row.get("metrics.Avg_ROUGE_L", 0)

                bleu_scores.append(bleu)
                rouge_scores.append(rouge)

                logger.debug(f"Run {label}: BLEU={bleu:.4f}, ROUGE-L={rouge:.4f}")

                tooltip = f"<b>Run ID:</b> {run_id}<br>"
                tooltip += f"<b>BLEU:</b> {bleu:.4f}<br>"
                tooltip += f"<b>ROUGE-L:</b> {rouge:.4f}<br>"

                try:
                    child_runs = mlflow.search_runs(
                        experiment_ids=[row["experiment_id"]],
                        filter_string=f"tags.mlflow.parentRunId = '{row['run_id']}'"
                    )

                    config_run = child_runs[child_runs["tags.mlflow.runName"] == "configurable_parameters"]

                    if not config_run.empty:
                        config_params = config_run.iloc[0].filter(like="params.")
                        if not config_params.empty:
                            tooltip += "<br><b>Configurable Parameters:</b><br>"
                            tooltip += "<br>".join(
                                f"<b>{escape(k.replace('params.', ''))}</b>: {escape(str(v))}"
                                for k, v in config_params.items()
                            )
                            logger.debug(f"Added {len(config_params)} parameters to tooltip for {label}")
                    else:
                        tooltip += "<br><i>No configurable parameters found</i>"
                        logger.debug(f"No configurable parameters found for {label}")
                except Exception as e:
                    tooltip += f"<br><i>Error loading parameters: {e}</i>"
                    logger.warning(f"Error loading parameters for {label}: {e}")

                tooltips.append(tooltip)

            logger.info("Creating Plotly figure...")
            fig = go.Figure()

            # Add invisible scatter for hover
            max_score = max(safe_max(bleu_scores), safe_max(rouge_scores))
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=[max_score * 1.05] * len(x_vals),
                mode='markers',
                marker=dict(size=15, color='rgba(0,0,0,0)'),
                hoverinfo='text',
                hovertext=tooltips,
                showlegend=False,
            ))

            # Add metric traces
            for scores, name, color in zip(
                    [bleu_scores, rouge_scores],
                    ["Avg BLEU", "Avg ROUGE-L"],
                    ["blue", "green"]
            ):
                fig.add_trace(go.Scatter(
                    x=x_vals, y=scores,
                    mode='lines+markers',
                    name=name,
                    marker=dict(color=color),
                    line=dict(color=color)
                ))
                logger.debug(f"Added {name} trace with {len(scores)} data points")

            fig.update_layout(
                title="Evaluation Metrics per MLflow Run",
                yaxis_title="Score",
                xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_texts, tickangle=-45),
                hovermode="x unified",
                template="plotly_white",
                margin=dict(b=150)
            )

            output_dir = "/output" if os.path.exists("/output") else "."
            html_path = os.path.join(output_dir, "run_metrics_interactive.html")
            logger.info(f"Saving plot to: {html_path}")

            with open(html_path, "w", encoding='utf-8') as f:
                f.write(pio.to_html(fig, full_html=True, include_plotlyjs="cdn"))

            try:
                mlflow.log_artifact(html_path)
                logger.info("Interactive plot saved and logged as MLflow artifact")
            except Exception as e:
                logger.error(f"Error logging plot artifact: {e}")

    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        logger.exception("Plot generation exception traceback:")


if __name__ == "__main__":
    main()