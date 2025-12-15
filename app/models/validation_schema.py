"""Pydantic models for validation ruleset schema - FIXED."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import uuid


# ============================================================================
# LLM RESPONSE SCHEMAS (for structured output from Gemini/OpenAI)
# ============================================================================

class LLMValidationRule(BaseModel):
    """Simplified rule schema for LLM extraction (Gemini structured output)."""
    rule: str = Field(
        description="The full sentence(s) describing the requirement, recommendation, or expectation."
    )
    pretext: str = Field(
        description="1-2 sentences explaining when/where this rule applies, in plain language."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Array of tags (leave empty, will be populated later)."
    )


class LLMRuleExtractionResponse(BaseModel):
    """Response schema for LLM rule extraction."""
    rules: List[LLMValidationRule] = Field(
        default_factory=list,
        description="List of all validation rules found on this page."
    )


# ============================================================================
# FULL DOCUMENT SCHEMAS (for MongoDB storage with tags)
# ============================================================================

class Tag(BaseModel):
    """Tag schema for rule classification."""
    category: str
    globalCategory: str
    mergeType: Literal["strict", "relaxed"] = "relaxed"
    lowerLimit: Optional[float] = None
    upperLimit: Optional[float] = None
    booleanValue: Optional[bool] = None
    type: Literal["Range", "Boolean", "Categorical"]
    main_tag: str = Field(alias="main-tag")
    
    model_config = ConfigDict(populate_by_name=True)


class ValidationRule(BaseModel):
    """Individual validation rule schema (with full tags)."""
    page_number: str
    rule: str
    pretext: str
    tags: List[Tag]  # Full Tag objects for final output
    rule_id: str = Field(default_factory=lambda: f"rid-{uuid.uuid4()}")


class MetaData(BaseModel):
    """Document metadata schema."""
    title: str
    description: str = ""
    category: str = "Guidelines"
    programId: Optional[str] = None
    trialId: Optional[str] = None
    tags: List[str] = []
    s3Key: str
    s3Uri: str
    fileType: str = "pdf"
    source: str = "system"
    disease: str = "general"


class ValidationDocument(BaseModel):
    """Complete validation document schema for MongoDB."""
    validation_ruleset: List[ValidationRule]
    s3_url: str
    state: str = "PROCESSING_RULES"
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    meta_data : MetaData
    
    model_config = ConfigDict(populate_by_name=True)
