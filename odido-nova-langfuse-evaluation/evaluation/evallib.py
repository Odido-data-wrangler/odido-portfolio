"""
Nova Evaluation Library (showcase)

This module provides evaluation functions for Nova's 3-agent workflow.
Adapted from HR chatbot evaluation architecture using proven LLM-as-judge patterns
with modular evaluation templates, structured feedback, and automated scoring.

Note: This is presentation code; dependencies on internal modules/secrets remain on purpose
and the file is not intended to run in this public repository.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import boto3  # type: ignore
import requests  # type: ignore
from langfuse import Langfuse  # type: ignore
from openai import AzureOpenAI  # type: ignore
from pydantic import BaseModel, Field

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
	import tomllib  # type: ignore
except ModuleNotFoundError:
	try:
		import tomli as tomllib  # type: ignore
	except ModuleNotFoundError:
		tomllib = None  # type: ignore

# Import Nova's existing configuration (placeholder path)
nova_src = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(nova_src))

from config.loaders import initialize_observability  # type: ignore

# Set up logging
logger = logging.getLogger(__name__)


# 1. Obtain Evaluation Templates from Config File
class EvalTemplates:
	def __init__(self) -> None:
		self.config_path = Path(__file__).parent / "configuration" / "general.toml"
		with open(self.config_path, "rb") as f:
			self.config = tomllib.load(f)  # type: ignore[arg-type]
		self.templates = self.config["experiment"]["evaluators"]

	def get_template(self, template_name: str) -> Optional[str]:
		return self.templates.get(template_name)  # type: ignore[no-any-return]


# 2. Specify Response Structure for Evaluation
class Feedback(BaseModel):
	comment: str = Field(
		description="Concise feedback in a single sentence. Break down the score into specific aspects of the response.",
	)
	score: float = Field(
		default=0.0,
		description="Score for the response between 0.0 and 1.0, where 1.0 is the best score.",
	)


user_prompt_template = """<text>
    answer: {answer}
    question: {question}
    context: {context}
    </text>"""


# 3. Obtain Evaluation Model
class LLM_access:
	def __init__(self, llm: str) -> None:
		self.llm_name = llm

	def get_available_llms(self) -> Dict[str, Callable]:
		"""Return the dictionary of available LLMs"""
		return {
			"gpt": self.get_gpt,
		}

	def get_llm(self) -> Optional[Callable]:
		llms = self.get_available_llms()
		return llms.get(self.llm_name)

	def get_gpt(
		self,
		eval_template: str,
		question: str,
		answer: str,
		context: Union[str, List[str]],
	) -> Union[Feedback, str]:
		try:
			# Use Nova's existing Azure OpenAI credentials from AWS Secrets Manager
			import sys as _sys
			from pathlib import Path as _Path

			# Add src to path for importing Nova's settings
			nova_src = _Path(__file__).parent.parent / 'src'
			if str(nova_src) not in _sys.path:
				_sys.path.insert(0, str(nova_src))

			# Import Nova's settings (with proper error handling)
			try:
				from config.schemas import Settings  # type: ignore

				# Initialize settings to get Azure OpenAI credentials from AWS Secrets
				settings = Settings()

				# Get required Azure OpenAI configuration
				endpoint = getattr(settings, 'azure_openai_endpoint', None)
				deployment = getattr(settings, 'azure_openai_deployment', None)
				subscription_key = getattr(settings, 'azure_openai_api_key', None)
				api_version = getattr(settings, 'azure_openai_api_version', '2025-01-01-preview')

				if not all([endpoint, deployment, subscription_key]):
					raise ValueError("Missing Azure OpenAI configuration from Nova's AWS Secrets Manager")

			except Exception as import_error:
				# Fallback to environment variables if Nova settings unavailable
				endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
				deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
				subscription_key = os.getenv('AZURE_OPENAI_API_KEY')
				api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')

				if not all([endpoint, deployment, subscription_key]):
					raise ValueError(
						f"Azure OpenAI credentials not available from Nova settings ({import_error}) "
						"or environment variables. Check AWS Secrets Manager configuration."
					)

			# Create Azure OpenAI client
			client = AzureOpenAI(
				api_key=subscription_key,  # type: ignore[arg-type]
				api_version=str(api_version),
				azure_endpoint=str(endpoint),
			)

			# Format the prompt
			user_prompt = user_prompt_template.format(
				answer=answer, question=question, context=context
			)
			system_prompt = EvalTemplates().get_template(eval_template)
			if system_prompt is None:
				raise ValueError(f"Evaluation template '{eval_template}' not found")

			# Make API call
			completion = client.chat.completions.create(
				model=str(deployment),
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				response_format={"type": "json_object"},
				max_tokens=512,
				temperature=0.0,
			)

			content = completion.choices[0].message.content
			if content is None:
				raise ValueError("Response content is None. Check the response format.")

			# Parse JSON response directly
			try:
				response_data = json.loads(content)
				return Feedback.model_validate(response_data)
			except json.JSONDecodeError:
				# Try to extract JSON from fenced code blocks
				import re
				m = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
				if m:
					return Feedback.model_validate(json.loads(m.group(1)))
				return Feedback.model_validate(json.loads(content))
		except Exception as e:  # pragma: no cover
			return f"Error: {e}"


# 4. Obtaining answers from Nova

def nova_production(question: str, intent: Optional[str] = None) -> Tuple[str, List[str], float]:
	"""
	Function to get responses from PRODUCTION Nova using direct AgentWorkflow execution.
	Generates full workflow traces in Langfuse automatically via LlamaIndex instrumentation.
	"""
	import asyncio as _asyncio
	from .workflow_evaluation import nova_workflow_production  # type: ignore
	return _asyncio.run(nova_workflow_production(question, intent))


def nova_development(question: str, intent: Optional[str] = None) -> Tuple[str, List[str], float]:
	"""
	Function to get responses from DEVELOPMENT Nova using direct AgentWorkflow execution.
	Generates full workflow traces in Langfuse automatically via LlamaIndex instrumentation.
	"""
	import asyncio as _asyncio
	from .workflow_evaluation import nova_workflow_development  # type: ignore
	return _asyncio.run(nova_workflow_development(question, intent))


def _nova_http_call(question: str, intent: Optional[str] = None, env: str = "production") -> Tuple[str, List[str], float]:
	"""Helper function to make HTTP request to Nova API (placeholder)."""
	try:
		from .config import Environment, NovaEndpoints  # type: ignore
		environment = Environment(env.lower())
		url = NovaEndpoints.get_endpoint(environment)
		if not url:
			raise ValueError(f"Nova endpoint not configured for environment: {env}.")

		payload = {"question": question, "user_id": "evaluation_user", "conversation_id": f"eval_{int(time.time())}"}
		if intent:
			payload["intent"] = intent
		resp = requests.post(url, json=payload, timeout=30)
		resp.raise_for_status()
		data = resp.json()
		answer = data.get("answer", "")
		context = data.get("context", [])
		retrieval_score = data.get("score", 0.8)
		return answer, context, retrieval_score
	except Exception as e:
		return f"Nova API call failed ({env}): {str(e)}", [f"Error context: {str(e)}"], 0.0


def create_testing_dataset(langfuse: Langfuse, dataset_name: str, mode: str = "dev") -> Dict[str, Tuple[str, List[str], float]]:
	"""Generate answers for a Langfuse dataset using Nova's workflow (showcase)."""
	import asyncio as _asyncio
	from .nova_workflow_test import create_workflow_testing_dataset  # type: ignore
	return _asyncio.run(create_workflow_testing_dataset(langfuse, dataset_name, mode))


# 5. Experiment Tools
async def create_benchmark(benchmark_config: Dict[str, Any], dataset_questions: List[Dict[str, Any]]) -> None:
	"""Create dataset and run initial V0 baseline evaluation (showcase)."""
	dataset_name = benchmark_config["experiment"]["dataset_name"]
	dataset_description = benchmark_config["experiment"]["dataset_description"]
	benchmark_run_name = benchmark_config["experiment"]["experiment_name"]

	langfuse = get_nova_langfuse_client()
	logger.info(f"Creating benchmark dataset: {dataset_name}")
	create_nova_dataset_from_test_cases(langfuse, dataset_name=dataset_name, description=dataset_description)
	logger.info("✅ Benchmark dataset created")
	evaluation_config = {"evaluations": benchmark_config["experiment"]["evaluations"], "model": benchmark_config["experiment"]["eval_model"], "temperature": 0.0}
	await run_nova_dataset_experiment(langfuse, dataset_name=dataset_name, run_name=benchmark_run_name, evaluation_config=evaluation_config)
	logger.info("🎉 Benchmark evaluated; see Langfuse")


async def run_experiment(
	langfuse: Langfuse,
	experiment_config: Dict[str, Any],
	test_dataset: Optional[Dict[str, Tuple[str, List[str], float]]] = None,
) -> None:
	"""Run experiment using dataset runs (showcase)."""
	dataset_name = experiment_config["experiment"]["dataset_name"]
	experiment_name = experiment_config["experiment"]["experiment_name"]
	evaluation_config = {"evaluations": experiment_config["experiment"]["evaluations"], "model": experiment_config["experiment"]["eval_model"], "temperature": 0.0}
	logger.info(f"Running experiment '{experiment_name}' on dataset '{dataset_name}'")
	if test_dataset:
		await run_experiment_with_test_dataset(langfuse, experiment_config, test_dataset)
	else:
		await run_nova_dataset_experiment(langfuse, dataset_name=dataset_name, run_name=experiment_name, evaluation_config=evaluation_config)
	logger.info("Experiment completed")


async def run_experiment_with_test_dataset(
	langfuse: Langfuse,
	experiment_config: Dict[str, Any],
	test_dataset: Dict[str, Tuple[str, List[str], float]],
) -> None:
	"""Legacy mode using a provided test dataset (showcase)."""
	dataset_name = experiment_config["experiment"]["dataset_name"]
	experiment_name = experiment_config["experiment"]["experiment_name"]
	evaluations = experiment_config["experiment"]["evaluations"]
	eval_model = experiment_config["experiment"]["eval_model"]
	try:
		dataset = langfuse.get_dataset(dataset_name)
	except Exception as e:
		raise Exception(f"Dataset '{dataset_name}' not found: {e}")
	llm_access = LLM_access(eval_model)
	eval_func = llm_access.get_llm()
	if eval_func is None:
		raise Exception(f"Unsupported LLM: {eval_model}")
	for item in dataset.items:
		question = item.input
		if question not in test_dataset:
			logger.warning(f"Question '{question}' not found in test dataset")
			continue
		answer, context, retrieval_score = test_dataset[question]
		with item.run(run_name=experiment_name, run_description=f"Legacy experiment run for {experiment_name}") as root_span:
			try:
				for eval_name in evaluations:
					template = EvalTemplates().get_template(eval_name)
					if not template:
						logger.error(f"Template not found for {eval_name}")
						continue
					output = eval_func(eval_template=eval_name, question=question, answer=answer, context=context)
					if hasattr(output, "score"):
						root_span.score_trace(name=eval_name, value=output.score, comment=output.comment)
					else:
						logger.error(f"Invalid output format for {eval_name}: {output}")
				root_span.score_trace(name="retrieval_score", value=retrieval_score)
			except Exception as e:
				logger.error(f"Error running evaluations: {e}")
	langfuse.flush()


# Helper function to get Nova's Langfuse instance

def get_nova_langfuse() -> Langfuse:
	"""Get Nova's existing Langfuse client (showcase)."""
	try:
		observability = initialize_observability()
		if not observability or 'langfuse' not in observability:
			raise Exception("Nova Langfuse configuration not available")
		return observability['langfuse']  # type: ignore[index]
	except Exception as e:
		# Fallback: build client from Settings
		from config.schemas import Settings  # type: ignore
		settings = Settings()
		public_key = getattr(settings, 'langfuse_public_key', None)
		secret_key = getattr(settings, 'langfuse_secret_key', None)
		host = getattr(settings, 'langfuse_host', None)
		if not all([public_key, secret_key, host]):
			raise Exception(f"Failed to get Nova Langfuse configuration: {e}")
		return Langfuse(host=host, public_key=public_key, secret_key=secret_key)  # type: ignore[arg-type]


# Enhanced Nova functions for dataset experiments

def get_nova_langfuse_client() -> Langfuse:
	"""Get Langfuse client using Nova's credentials from AWS Secrets Manager (showcase)."""
	try:
		client = boto3.client('secretsmanager', region_name='eu-central-1')
		response = client.get_secret_value(SecretId='ds-secret-dev-nova-application-credentials')
		secret_data = json.loads(response['SecretString'])
		return Langfuse(host=secret_data['langfuse_host'], public_key=secret_data['langfuse_public_key'], secret_key=secret_data['langfuse_secret_key'])
	except Exception as e:
		logger.error(f"Failed to get Nova Langfuse client: {e}")
		raise


async def nova_query_function(question: str) -> Dict[str, Any]:
	"""Query Nova using the external WebSocket approach (showcase)."""
	try:
		from nova_external_test import ask_nova_external_evaluation  # type: ignore
		result = ask_nova_external_evaluation(question, skip_evaluations=True)
		return {
			"success": result.get("success", False),
			"answer": result.get("answer", ""),
			"intent": result.get("intent", ""),
			"sources": result.get("sources", []),
			"error": result.get("error", ""),
		}
	except Exception as e:
		logger.error(f"Nova query failed for '{question}': {e}")
		return {"success": False, "answer": "", "intent": "", "sources": [], "error": str(e)}


async def evaluator_function(
	eval_name: str,
	question: str,
	answer: str,
	expected_answer: str,
	context: List[str],
	intent: str,
) -> Optional[Dict[str, Any]]:
	"""Evaluate Nova responses using Azure OpenAI (showcase)."""
	try:
		from nova_external_test import run_single_evaluation, get_azure_openai_client  # type: ignore
		azure_client = get_azure_openai_client()
		context_str = "\n".join([f"Source {i+1}: {str(s)}" for i, s in enumerate(context)]) if context else "No sources retrieved"
		fb = run_single_evaluation(eval_name, question, answer, context_str, azure_client)
		return {"score": fb.score, "comment": fb.comment}
	except Exception as e:
		logger.error(f"Evaluation {eval_name} failed: {e}")
		return None


def create_nova_dataset_from_test_cases(
	langfuse_client: Langfuse,
	dataset_name: str = "nova_evaluation_dataset",
	description: Optional[str] = None,
) -> str:
	"""Create a Langfuse dataset from tests/data/test_cases.json (showcase)."""
	if description is None:
		description = f"Nova evaluation dataset created from test_cases.json on {datetime.now().isoformat()}"
	test_cases_path = Path(__file__).parent.parent / "tests" / "data" / "test_cases.json"
	logger.info(f"Creating dataset '{dataset_name}' from {test_cases_path}")
	with open(test_cases_path, 'r', encoding='utf-8') as f:
		test_cases = json.load(f)
	langfuse_client.create_dataset(name=dataset_name, description=description)
	for i, tc in enumerate(test_cases):
		try:
			metadata = {
				"intent": tc.get("intent", "unknown"),
				"knowledge_base": tc.get("knowledge_base", "general"),
				"expected_sources": tc.get("expected_sources", []),
				"answer_quality": tc.get("answer_quality", ""),
				"follow_up": tc.get("follow_up", "No"),
				"test_id": tc.get("id", f"test_{i:03d}"),
			}
			langfuse_client.create_dataset_item(dataset_name=dataset_name, input=tc["question"], expected_output=tc.get("expected_answer", ""), metadata=metadata)
		except Exception as e:
			logger.error(f"Failed to add test case {tc.get('id', i)}: {e}")
	logger.info(f"Dataset '{dataset_name}' created with {len(test_cases)} items")
	return dataset_name


async def run_nova_dataset_experiment(
	langfuse_client: Langfuse,
	dataset_name: str,
	run_name: str,
	evaluation_config: Optional[Dict[str, Any]] = None,
) -> str:
	"""Run a complete dataset experiment using dataset runs (showcase)."""
	logger.info(f"Starting dataset experiment '{run_name}' on dataset '{dataset_name}'")
	try:
		dataset = langfuse_client.get_dataset(dataset_name)
	except Exception as e:
		logger.error(f"Dataset '{dataset_name}' not found: {e}")
		raise
	results: List[Dict[str, Any]] = []
	for i, item in enumerate(dataset.items, 1):
		try:
			with item.run(run_name=run_name, run_description=f"Nova evaluation run on {datetime.now().isoformat()}") as root_span:
				question = item.input
				expected_output = item.expected_output or ""
				nova_result = await nova_query_function(question)
				if not nova_result.get("success", False):
					results.append({"success": False, "error": nova_result.get("error", "")})
					continue
				answer = nova_result.get("answer", "")
				intent = nova_result.get("intent", "")
				sources = nova_result.get("sources", [])
				evaluation_results: Dict[str, Any] = {}
				if evaluation_config:
					for name in evaluation_config.get("evaluations", ["relevance", "accuracy", "helpfulness", "language_quality", "intent_recognition"]):
						try:
							res = await evaluator_function(name, question, answer, expected_output, sources, intent)
							if res:
								root_span.score_trace(name=name, value=res.get("score", 0.0), comment=res.get("comment", ""))
								evaluation_results[name] = res
						except Exception as e:
							logger.error(f"Evaluation {name} failed: {e}")
				root_span.score_trace(name="nova_success", value=1.0 if answer else 0.0, comment=f"Intent: {intent}, Sources: {len(sources)}")
				results.append({"success": True, "item_id": getattr(item, 'id', f'item_{i}'), "answer": answer, "intent": intent, "sources_count": len(sources), "evaluation_results": evaluation_results})
		except Exception as e:
			logger.error(f"Failed to process item {getattr(item, 'id', f'item_{i}')}: {e}")
			results.append({"success": False, "error": str(e), "item_id": getattr(item, 'id', f'item_{i}')})
	langfuse_client.flush()
	logger.info(f"Dataset experiment completed. Run: {run_name}")
	return run_name
