from typing import Any, Dict, List, Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field # This may also just be pydatic (no v1, see argument description for BaseTool)
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import (
    RerankConfig,
    SummaryConfig,
    VectaraQueryConfig
)

class VectaraInput(BaseModel):
    "Input for the Vectara query tool."

    query: str = Field(description="The input query from the user")


class VectaraRAG(BaseTool):
    "Tool that queries a Vectara corpus."

    name: str = "vectara_rag"
    description: str = (
        "A wrapper around Vectara. "
        "Useful when you want to search documents for information. "
        "Input should be a user query."
    )


    def __init__(
            self,
            vectara_customer_id: str,
            vectara_corpus_id: str,
            vectara_api_key: str,
            args_schema: Type[BaseModel] = VectaraInput,
            summarizer_prompt_name: str = "vectara-summary-ext-24-05-sml",
            summary_num_results: int = 5,
            summary_response_lang: str = "eng",
            num_results: int = 5,
            n_sentences_before: int = 2,
            n_sentences_after: int = 2,
            metadata_filter: str = "",
            lambda_val: float = 0.005,
            reranker: str = "mmr",
            rerank_k: int = 50,
            mmr_diversity_bias: float = 0.2,
            include_citations: bool = False,
    ) -> None:
        """
        Initializes the Vectara API and query parameters.

        Parameters:
        - vectara_customer_id (str): Your Vectara customer ID.
        - vectara_corpus_id (str): The corpus ID for the corpus you want to search for information.
        - vectara_api_key (str): An API key that has query permissions for the given corpus.
        - args_schema (BaseModel): The argument schema for the query tool
        - summarizer_prompt_name (str): If enable_summarizer is True, the Vectara summarizer to use.
        - summary_num_results (int): If enable_summarizer is True, the number of summary results.
        - summary_response_lang (str): If enable_summarizer is True, the response language for the summary.
        - num_results (int): Number of search results to return with response.
        - n_sentences_before (int): Number of sentences before the summary.
        - n_sentences_after (int): Number of sentences after the summary.
        - metadata_filter (str): A string with expressions to filter the search documents.
        - lambda_val (float): Lambda value for the Vectara query.
        - reranker (str): The reranker type, either "mmr", "rerank_multilingual_v1" or "none".
        - rerank_k (int): Number of top-k documents for reranking.
        - mmr_diversity_bias (float): MMR diversity bias.
        - include_citations (bool): Whether to include citations in the response.
          If True, uses MARKDOWN vectara citations that requires the Vectara scale plan. # MAY OR MAY NOT NEED THIS LINE OF COMMENT
        """

        self.args_schema = args_schema
        self.include_citations = include_citations

        vectara = Vectara(
            vectara_customer_id = vectara_customer_id,
            vectara_corpus_id=vectara_corpus_id,
            vectara_api_key=vectara_api_key,
        )

        summary_config = SummaryConfig(
            is_enabled=True,
            max_results = summary_num_results,
            prompt_name=summarizer_prompt_name,
            response_lang=summary_response_lang,
        )

        rerank_config = RerankConfig(
            reranker=reranker,
            rerank_k = rerank_k,
            mmr_diversity_bias=mmr_diversity_bias,
        )

        query_config = VectaraQueryConfig(
            k = num_results,
            lambda_val = lambda_val,
            filter = metadata_filter,
            n_sentence_before = n_sentences_before,
            n_sentence_after = n_sentences_after,
            rerank_config = rerank_config,
            summary_config = summary_config,
        )

        self.rag = vectara.as_rag(query_config)


    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None, # NOT SURE WHAT THIS IS FOR
    ) -> Dict[str, Any]:
        """Use the tool to answer user's query with the corpus documents."""
        
        # NEED TO FIGURE OUT HOW TO IMPLEMENT ARGS SCHEMA

        response = self.rag.invoke(query)
        
        if self.include_citations:
            return {
                "summary": response['answer'],
                "citations": response['context']
            }
        else:
            return {
                "summary": response['answer']
            }
        


class VectaraRetriever(BaseTool):
    "Tool that retrieves the top search results from a Vectara corpus based on a user query."

    name: str = "vectara_retriever"
    description: str = (
        "A wrapper around Vectara. "
        "Useful when you want to search documents for information. "
        "Input should be a user query."
    )


    def __init__(
            self,
            vectara_customer_id: str,
            vectara_corpus_id: str,
            vectara_api_key: str,
            args_schema: Type[BaseModel] = VectaraInput,
            num_results: int = 5,
            n_sentences_before: int = 2,
            n_sentences_after: int = 2,
            metadata_filter: str = "",
            lambda_val: float = 0.005,
            reranker: str = "mmr",
            rerank_k: int = 50,
            mmr_diversity_bias: float = 0.2,
    ) -> None:
        """
        Initializes the Vectara API and query parameters.

        Parameters:
        - vectara_customer_id (str): Your Vectara customer ID.
        - vectara_corpus_id (str): The corpus ID for the corpus you want to search for information.
        - vectara_api_key (str): An API key that has query permissions for the given corpus.
        - args_schema (BaseModel): The argument schema for the query tool
        - num_results (int): Number of search results to return from tool.
        - n_sentences_before (int): Number of sentences before the summary.
        - n_sentences_after (int): Number of sentences after the summary.
        - metadata_filter (str): A string with expressions to filter the search documents.
        - lambda_val (float): Lambda value for the Vectara query.
        - reranker (str): The reranker type, either "mmr", "rerank_multilingual_v1" or "none".
        - rerank_k (int): Number of top-k documents for reranking.
        - mmr_diversity_bias (float): MMR diversity bias.
        """

        self.args_schema = args_schema

        vectara = Vectara(
            vectara_customer_id = vectara_customer_id,
            vectara_corpus_id=vectara_corpus_id,
            vectara_api_key=vectara_api_key,
        )

        rerank_config = RerankConfig(
            reranker=reranker,
            rerank_k = rerank_k,
            mmr_diversity_bias=mmr_diversity_bias,
        )

        query_config = VectaraQueryConfig(
            k = num_results,
            lambda_val = lambda_val,
            filter = metadata_filter,
            n_sentence_before = n_sentences_before,
            n_sentence_after = n_sentences_after,
            rerank_config = rerank_config,
        )

        self.retriever = vectara.as_retriever(config=query_config)


    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None, # NOT SURE WHAT THIS IS FOR
    ) -> List[Dict]:
        """Use the tool to retrieve the top search results based on the user's query."""
        
        response = self.retriever._get_relevant_documents(query=query, run_manager = run_manager)
        
        return [{"text": doc.page_content, "metadata": doc.metadata} for doc in response]