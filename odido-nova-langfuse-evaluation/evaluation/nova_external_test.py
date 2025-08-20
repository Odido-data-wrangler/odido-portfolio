#!/usr/bin/env python3
"""
Nova External Evaluation - Self-Contained Pattern (showcase)

This file demonstrates how I generated Langfuse traces for Nova using a
self-contained pattern:
- No internal Nova imports
- WebSocket integration
- LLM-as-judge scoring with JSON outputs
- Langfuse scoring on the current trace

Important:
- This is presentation code. It is not intended to run in this public repo.
- All secrets/endpoints are placeholders; remove or replace before any execution.
"""

from __future__ import annotations

import json
import os
import ssl
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3  # type: ignore
import websocket  # type: ignore
from langfuse import Langfuse  # type: ignore
from openai import AzureOpenAI  # type: ignore
from pydantic import BaseModel, Field

# Optional TOML loader (py311 tomllib, else tomli)
try:
	import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
	try:
		import tomli as tomllib  # type: ignore
	except ModuleNotFoundError:
		tomllib = None  # type: ignore

# --------------------------------------------------------------------------------------
# Config (sanitized placeholders)
# --------------------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
AWS_WS_HOST = os.getenv("NOVA_WS_HOST", "REDACTED_NOVA_WS_HOST")  # e.g. nova_test.dev01... (placeholder)
AWS_PEM_SECRET_ARN = os.getenv(
	"NOVA_PEM_SECRET_ARN",
	"arn:aws:secretsmanager:eu-central-1:000000000000:secret:REDACTED",
)
USER_ID = os.getenv("NOVA_USER_ID", "demo_user@example.com")

# Optional: names for secrets; left as placeholders on purpose
SECRET_APP_CREDENTIALS = os.getenv("NOVA_APP_SECRETS", "REDACTED_APP_SECRETS_NAME")
SECRET_AZURE_OPENAI = os.getenv("NOVA_AOAI_SECRETS", "REDACTED_AOAI_SECRETS_NAME")


# --------------------------------------------------------------------------------------
# Models and templates
# --------------------------------------------------------------------------------------
class EvaluationFeedback(BaseModel):
	"""Structured feedback used by LLM-as-judge."""
	comment: str = Field(description="Concise feedback explaining the score")
	score: float = Field(description="Score between 0.0 and 1.0")


def _load_evaluator_templates() -> Dict[str, str]:
	"""Load evaluator templates from configuration, fallback to minimal inline.
	This mirrors the pattern I used on the internal setup.
	"""
	# Try general.toml next to this file: evaluation/configuration/general.toml
	if tomllib is not None:
		config_path = Path(__file__).parent / "configuration" / "general.toml"
		if config_path.exists():
			try:
				with open(config_path, "rb") as f:
					cfg = tomllib.load(f)
				return dict(cfg.get("experiment", {}).get("evaluators", {}))  # type: ignore
			except Exception:
				pass
	# Fallback minimal template set
	return {
		"relevance": "Return JSON {\"score\": float, \"comment\": string} about relevance.",
		"completeness": "Return JSON for completeness.",
		"accuracy": "Return JSON for accuracy; penalize unsupported claims.",
		"helpfulness": "Return JSON for helpfulness.",
		"language_quality": "Return JSON for Dutch language quality.",
		"intent_recognition": "Return JSON for correct intent domain.",
	}


EVALUATION_TEMPLATES = _load_evaluator_templates()


# --------------------------------------------------------------------------------------
# Secrets helpers (placeholders; non-functional in public repo)
# --------------------------------------------------------------------------------------

def _get_secrets_client():
	"""Create a Secrets Manager client using default AWS provider chain."""
	return boto3.client("secretsmanager", region_name=AWS_REGION)


def get_credentials() -> tuple[Dict[str, Any], str]:
	"""Fetch Langfuse creds and a PEM cert from Secrets Manager (names redacted).
	In public repo this returns placeholders or raises if secrets are unavailable.
	"""
	client = _get_secrets_client()
	# App credentials (Langfuse host/public/secret, ws token etc.)
	app = client.get_secret_value(SecretId=SECRET_APP_CREDENTIALS)
	app_data = json.loads(app["SecretString"])  # type: ignore[index]
	# PEM certificate for WS (optional depending on env)
	pem = client.get_secret_value(SecretId=AWS_PEM_SECRET_ARN)
	pem_content = pem["SecretString"]  # type: ignore[index]
	return app_data, pem_content  # type: ignore[return-value]


def write_temp_pem(content: str) -> str:
	"""Write PEM to a secure temp file; return path; caller must delete."""
	fd, path = tempfile.mkstemp()
	try:
		with os.fdopen(fd, "w") as tmp:
			tmp.write(content)
		os.chmod(path, 0o600)
	except Exception:
		os.remove(path)
		raise
	return path


def get_nova_token() -> str:
	"""Retrieve WS token from app secrets (placeholder)."""
	client = _get_secrets_client()
	app = client.get_secret_value(SecretId=SECRET_APP_CREDENTIALS)
	data = json.loads(app["SecretString"])  # type: ignore[index]
	return data.get("ws_auth_token", "REDACTED_TOKEN")


def get_azure_openai_client() -> AzureOpenAI:
	"""Create Azure OpenAI client from secrets (placeholder names)."""
	client = _get_secrets_client()
	resp = client.get_secret_value(SecretId=SECRET_AZURE_OPENAI)
	secret = json.loads(resp["SecretString"])  # type: ignore[index]
	return AzureOpenAI(
		api_key=secret.get("api_key") or secret.get("azure_openai_api_key"),
		api_version=secret.get("api_version", secret.get("azure_openai_api_version", "2024-10-21")),
		azure_endpoint=secret.get("endpoint", secret.get("azure_openai_endpoint")),
	)


# --------------------------------------------------------------------------------------
# LLM-as-judge
# --------------------------------------------------------------------------------------

def run_single_evaluation(
	eval_name: str,
	question: str,
	answer: str,
	context: str,
	azure_client: AzureOpenAI,
) -> EvaluationFeedback:
	"""Invoke Azure OpenAI with the selected evaluator template."""
	template = EVALUATION_TEMPLATES.get(eval_name)
	if not template:
		return EvaluationFeedback(score=0.0, comment=f"Template not found for {eval_name}")

	user_prompt = f"""<text>\n    answer: {answer}\n    question: {question}\n    context: {context}\n    </text>"""

	# deployment name comes from the same secret
	secrets_client = _get_secrets_client()
	resp = secrets_client.get_secret_value(SecretId=SECRET_AZURE_OPENAI)
	secret = json.loads(resp["SecretString"])  # type: ignore[index]
	deployment = secret.get("deployment", secret.get("azure_openai_deployment", "REDACTED_DEPLOYMENT"))

	completion = azure_client.chat.completions.create(
		model=deployment,
		messages=[
			{"role": "system", "content": template},
			{"role": "user", "content": user_prompt},
		],
		response_format={"type": "json_object"},
		max_tokens=512,
		temperature=0.0,
	)
	content = completion.choices[0].message.content or "{}"
	data = json.loads(content)
	return EvaluationFeedback.model_validate(data)


def run_all_evaluations(
	question: str,
	answer: str,
	sources: List[Dict[str, Any]],
	intent: Optional[str] = None,
) -> Dict[str, EvaluationFeedback]:
	"""Run all judge metrics and return a mapping."""
	azure_client = get_azure_openai_client()
	if sources:
		ctx_lines = []
		for i, src in enumerate(sources, 1):
			if isinstance(src, dict):
				ctx_lines.append(f"Source {i}: {src.get('source', 'unknown')} - {src.get('url', 'unknown')}")
			else:
				ctx_lines.append(f"Source {i}: {str(src)}")
		context = "\n".join(ctx_lines)
	else:
		context = "No sources retrieved"

	metrics = [
		"relevance",
		"completeness",
		"accuracy",
		"helpfulness",
		"language_quality",
		"intent_recognition",
	]
	results: Dict[str, EvaluationFeedback] = {}
	for name in metrics:
		fb = run_single_evaluation(name, question, answer, context, azure_client)
		results[name] = fb
	return results


# --------------------------------------------------------------------------------------
# Main external evaluation flow (WS + Langfuse scoring)
# --------------------------------------------------------------------------------------

def ask_nova_external_evaluation(
	question: str,
	conversation_id: Optional[str] = None,
	skip_evaluations: bool = False,
) -> Dict[str, Any]:
	"""Create Langfuse trace, connect to WS, collect results, score with judge.
	This mirrors the internal proof-of-concept used for screenshots.
	"""
	conversation_id = conversation_id or f"external_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

	# Fetch secrets (host/keys + PEM cert)
	lf_secrets, pem_content = get_credentials()
	pem_path = write_temp_pem(pem_content)

	langfuse = Langfuse(
		host=lf_secrets.get("langfuse_host", "https://REDACTED.langfuse"),
		public_key=lf_secrets.get("langfuse_public_key", "REDACTED"),
		secret_key=lf_secrets.get("langfuse_secret_key", "REDACTED"),
	)

	with langfuse.start_as_current_span(
		name="nova_external_evaluation",
		input={
			"question": question,
			"conversation_id": conversation_id,
			"timestamp": datetime.now().isoformat(),
			"evaluation_type": "external_pattern",
		},
	) as span:
		# Connect WS (cert validation intentionally disabled in showcase)
		ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
		token = get_nova_token()
		ws.connect(f"wss://{AWS_WS_HOST}/ws/query/{USER_ID}?token={token}")

		result: Dict[str, Any] = {
			"question": question,
			"conversation_id": conversation_id,
			"answer": "",
			"intent": None,
			"sources": [],
			"trace_id": span.trace_id,
			"success": False,
			"error": None,
			"evaluations": {},
			"evaluation_summary": {},
		}

		# Simplified receive loop (abbreviated for showcase)
		question_sent = False
		for i in range(50):
			if i == 0:
				ws.settimeout(5.0)
			else:
				ws.settimeout(30.0)
			try:
				text = ws.recv()
			except websocket.WebSocketTimeoutException:
				if i == 0 and not question_sent:
					ws.send(json.dumps({"type": "question", "payload": {"question": question, "conversation_id": conversation_id}}))
					question_sent = True
					continue
				break

			try:
				msg = json.loads(text)
			except json.JSONDecodeError:
				if len(text) > 0:
					result["answer"] += text
				continue

			typ = msg.get("type")
			if typ == "input_required":
				payload = msg.get("payload", {})
				cust = payload.get("customer_id")
				if cust:
					ws.send(json.dumps({"type": "confirm_customer", "payload": {"customer_id": cust, "confirmed": True}}))
					if not question_sent:
						ws.send(json.dumps({"type": "question", "payload": {"question": question, "conversation_id": conversation_id}}))
						question_sent = True
			elif typ == "tool_result":
				out = msg.get("payload", {}).get("output")
				if isinstance(out, dict) and "intent" in out:
					result["intent"] = out.get("intent")
			elif typ == "context_retrieved":
				result["sources"] = msg.get("sources", [])
			elif typ == "response_streaming":
				result["answer"] += msg.get("chunk", "")
			elif typ in {"final_result", "response_complete"}:
				payload = msg.get("payload", {}) if typ == "final_result" else {}
				if isinstance(payload, dict):
					result["answer"] = payload.get("final_answer", result["answer"]) or result["answer"]
					result["sources"] = payload.get("sources", result["sources"]) or result["sources"]
				result["success"] = True
				break
			elif typ == "error":
				result["error"] = msg.get("payload", "Unknown error")
				break

		ws.close()

		# Add judge scores
		if result["success"] and result["answer"] and not skip_evaluations:
			evals = run_all_evaluations(question, result["answer"], result["sources"], result["intent"])  # type: ignore[arg-type]
			result["evaluations"] = evals
			scores = {k: v.score for k, v in evals.items()}
			avg = sum(scores.values()) / len(scores) if scores else 0.0
			best = max(scores.items(), key=lambda x: x[1]) if scores else ("None", 0.0)
			worst = min(scores.items(), key=lambda x: x[1]) if scores else ("None", 0.0)
			result["evaluation_summary"] = {
				"total_evaluations": len(evals),
				"average_score": avg,
				"scores": scores,
				"best_evaluation": best,
				"worst_evaluation": worst,
			}
			for name, fb in evals.items():
				langfuse.score_current_trace(name=name, value=fb.score, comment=fb.comment)
		elif not result["success"]:
			langfuse.score_current_trace(name="nova_basic_quality", value=0.0)

		# Update trace output snapshot for the screenshots
		span.update(
			output={
				"answer": (result["answer"] or "")[:500],
				"intent": result["intent"],
				"sources_count": len(result["sources"]),
				"success": result["success"],
				"error": result["error"],
				"evaluations_completed": len(result.get("evaluations", {})),
				"average_evaluation_score": result.get("evaluation_summary", {}).get("average_score", 0.0),
				"evaluation_type": "external_pattern",
			}
		)

	# Cleanup
	os.remove(pem_path)
	langfuse.flush()
	return result


def run_external_evaluation_demo() -> List[Dict[str, Any]]:
	"""Small wrapper for the slides/screenshots; not intended to run here."""
	questions = ["Wat kost een mobiel abonnement?"]
	results: List[Dict[str, Any]] = []
	for q in questions:
		results.append(ask_nova_external_evaluation(q))
	return results


if __name__ == "__main__":  # pragma: no cover
	print("This showcase file is not meant to run in this public repo.")
