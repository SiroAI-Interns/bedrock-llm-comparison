# app/agents/research_agent.py
"""Agent 0: Research Agent - Gathers and synthesizes web search results."""

from typing import List, Dict, Optional
import re
from duckduckgo_search import DDGS
from app.core.unified_bedrock_client import UnifiedBedrockClient
from app.agents.templates import RESEARCH_SYNTHESIS_TEMPLATE


class ResearchAgent:
    """Performs web research and synthesizes factual information."""

    def __init__(self, research_client: UnifiedBedrockClient):
        """
        Initialize Research Agent.
        
        Args:
            research_client: Bedrock client for synthesizing research
        """
        self.research_client = research_client

    def _is_mostly_english(self, text: str) -> bool:
        """
        Check if text is mostly English (ASCII characters).
        
        Args:
            text: Text to check
        
        Returns:
            True if mostly English, False otherwise
        """
        if not text:
            return True
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return (ascii_count / len(text)) > 0.7

    def extract_search_query(self, user_prompt: str) -> str:
        """
        Extract key search terms from user prompt.
        Removes instructional language like "Explain", "Answer in X sentences", etc.
        
        Args:
            user_prompt: Original user query
        
        Returns:
            Cleaned search query with key terms
        """
        extraction_prompt = f"""Extract the core topic/keywords from this query for web searching.
Remove instructional words like "explain", "describe", "answer in X sentences", etc.
Return ONLY the key search terms, nothing else.

Query: {user_prompt}

Search terms:"""
        
        try:
            search_query = self.research_client.generate(extraction_prompt).strip()
            return search_query.strip('"').strip("'")
        except Exception as e:
            print(f"⚠️  Query extraction failed, using fallback: {e}")
            # Fallback: simple regex cleanup
            cleaned = re.sub(
                r'^(explain|describe|what is|tell me about|answer)\s+',
                '', user_prompt, flags=re.IGNORECASE
            )
            cleaned = re.sub(r'\.\s*answer in.*$', '', cleaned, flags=re.IGNORECASE)
            return cleaned.strip()

    def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform web search using DuckDuckGo (free, no API key needed).
        Returns English results only.
        
        Args:
            query: Search query
            max_results: Number of results to return
        
        Returns:
            List of dicts with 'title', 'url', 'snippet'
        """
        try:
            # Force English results by setting region
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=max_results * 2  # Get extra to filter
                ))
            
            # Filter for English content only
            english_results = []
            for r in results:
                title = r.get("title", "No title")
                snippet = r.get("body", "")
                
                if self._is_mostly_english(title) and self._is_mostly_english(snippet):
                    english_results.append({
                        "title": title,
                        "url": r.get("href", ""),
                        "snippet": snippet[:500],
                    })
                    if len(english_results) >= max_results:
                        break
            
            # Fallback if no English results
            if not english_results:
                print("⚠️  No English results found, using all results...")
                return [
                    {
                        "title": r.get("title", "No title"),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:500],
                    }
                    for r in results[:max_results]
                ]
            
            return english_results
            
        except Exception as e:
            print(f"⚠️  Web search error: {e}")
            return []

    def run(
        self,
        prompt: str,
        synthesis_template: Optional[str] = None
    ) -> Dict:
        """
        Run research agent to gather and synthesize information.
        
        Args:
            prompt: User's query to research
            synthesis_template: Custom template for synthesizing research
        
        Returns:
            Dictionary with research context and sources
        """
        print("\n" + "="*70)
        print("AGENT 0: RESEARCH AGENT")
        print("="*70)
        print(f"📚 Researching: {prompt[:100]}...\n")

        # Use default template if not provided
        if synthesis_template is None:
            synthesis_template = RESEARCH_SYNTHESIS_TEMPLATE

        # Extract key search terms
        print("🔍 Extracting search terms from query...")
        search_query = self.extract_search_query(prompt)
        print(f"📝 Search query: {search_query}\n")

        # Perform web search
        print("🌐 Searching the web for factual information...")
        search_results = self.web_search(search_query, max_results=5)

        if not search_results:
            print("⚠️  No search results found, proceeding without research context")
            return {
                "prompt": prompt,
                "research_context": "No web research available.",
                "sources": [],
                "raw_search_results": "",
            }

        print(f"✅ Found {len(search_results)} sources\n")

        # Format search results for synthesis
        search_text = ""
        sources = []
        
        for idx, result in enumerate(search_results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            
            search_text += f"\n[Source {idx}] {title}\n"
            search_text += f"URL: {url}\n"
            search_text += f"Content: {snippet}\n"
            search_text += "-" * 60 + "\n"
            
            sources.append({"index": idx, "title": title, "url": url})
            
            # Print source for visibility
            print(f"  [{idx}] {title[:70]}...")
            print(f"      {url}")

        # Build research prompt
        research_prompt = synthesis_template.format(
            query=prompt,
            search_results=search_text,
        )

        print("\n🤖 Synthesizing research findings...")
        research_context = self.research_client.generate(research_prompt)
        print(f"✅ Research complete - {len(sources)} sources gathered\n")

        return {
            "prompt": prompt,
            "research_context": research_context,
            "sources": sources,
            "raw_search_results": search_text,
        }
