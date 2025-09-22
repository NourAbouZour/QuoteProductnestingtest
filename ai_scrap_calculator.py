#!/usr/bin/env python3
"""
AI-Powered Scrap Calculation System
Uses OpenAI API for intelligent scrap factor prediction and optimization
"""

import time
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class ScrapCalculationInput:
    """Input data for scrap calculation"""
    parts_data: List[Dict[str, Any]]  # List of parts with dimensions and quantities
    boards_data: List[Dict[str, Any]]  # Available boards
    # Nesting functionality removed for deployment
    material_type: str = "steel"
    thickness_mm: float = 1.0
    complexity_score: float = 0.0

@dataclass
class ScrapCalculationResult:
    """Result of AI scrap calculation"""
    scrap_factor: float
    confidence_score: float
    reasoning: str
    calculation_time: float
    optimization_suggestions: List[str]
    estimated_utilization: float

class AIScrapCalculator:
    """AI-powered scrap factor calculator using OpenAI API"""
    
    def __init__(self, openai_api_key: Optional[str] = None, use_cache: bool = True):
        self.api_key = openai_api_key
        self.use_cache = use_cache
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.session = None
        
    async def initialize_session(self):
        """Initialize aiohttp session for API calls"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _calculate_complexity_score(self, parts_data: List[Dict[str, Any]]) -> float:
        """Calculate complexity score based on parts characteristics"""
        if not parts_data:
            return 0.0
        
        total_parts = sum(part.get('quantity', 1) for part in parts_data)
        unique_shapes = len(set((part.get('length_mm', 0), part.get('width_mm', 0)) for part in parts_data))
        
        # Calculate size variance
        areas = [part.get('length_mm', 0) * part.get('width_mm', 0) for part in parts_data]
        size_variance = np.var(areas) if len(areas) > 1 else 0
        
        # Complexity factors
        complexity = 0.0
        complexity += min(1.0, total_parts / 100) * 0.3  # Part count factor
        complexity += min(1.0, unique_shapes / 20) * 0.3  # Shape variety factor
        complexity += min(1.0, size_variance / 1000000) * 0.4  # Size variance factor
        
        return complexity
    
    def _create_cache_key(self, input_data: ScrapCalculationInput) -> str:
        """Create a cache key for the input data"""
        # Create a simplified hash of the input data
        key_data = {
            'parts_count': len(input_data.parts_data),
            'total_quantity': sum(part.get('quantity', 1) for part in input_data.parts_data),
            'material': input_data.material_type,
            'thickness': input_data.thickness_mm,
            'complexity': input_data.complexity_score
        }
        return json.dumps(key_data, sort_keys=True)
    
    async def calculate_scrap_factor_ai(self, input_data: ScrapCalculationInput) -> ScrapCalculationResult:
        """Calculate scrap factor using AI analysis"""
        start_time = time.time()
        
        # Check cache first
        if self.use_cache:
            cache_key = self._create_cache_key(input_data)
            with self.cache_lock:
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key]
                    cached_result.calculation_time = time.time() - start_time
                    return cached_result
        
        try:
            await self.initialize_session()
            
            # Prepare data for AI analysis
            analysis_data = self._prepare_ai_analysis_data(input_data)
            
            # Use OpenAI API for intelligent analysis
            if self.api_key:
                result = await self._analyze_with_openai(analysis_data)
            else:
                # Fallback to rule-based calculation
                result = await self._analyze_with_rules(analysis_data)
            
            # Cache the result
            if self.use_cache:
                cache_key = self._create_cache_key(input_data)
                with self.cache_lock:
                    self.cache[cache_key] = result
            
            result.calculation_time = time.time() - start_time
            return result
            
        except Exception as e:
            # Fallback to basic calculation
            return await self._fallback_calculation(input_data, start_time)
    
    def _prepare_ai_analysis_data(self, input_data: ScrapCalculationInput) -> Dict[str, Any]:
        """Prepare data for AI analysis"""
        parts_summary = {
            'total_parts': sum(part.get('quantity', 1) for part in input_data.parts_data),
            'unique_shapes': len(set((part.get('length_mm', 0), part.get('width_mm', 0)) for part in input_data.parts_data)),
            'size_range': {
                'min_area': min(part.get('length_mm', 0) * part.get('width_mm', 0) for part in input_data.parts_data),
                'max_area': max(part.get('length_mm', 0) * part.get('width_mm', 0) for part in input_data.parts_data),
                'avg_area': np.mean([part.get('length_mm', 0) * part.get('width_mm', 0) for part in input_data.parts_data])
            },
            'quantity_distribution': {
                'min_qty': min(part.get('quantity', 1) for part in input_data.parts_data),
                'max_qty': max(part.get('quantity', 1) for part in input_data.parts_data),
                'avg_qty': np.mean([part.get('quantity', 1) for part in input_data.parts_data])
            }
        }
        
        boards_summary = {
            'total_boards': len(input_data.boards_data),
            'size_range': {
                'min_area': min(board.get('width_mm', 0) * board.get('height_mm', 0) for board in input_data.boards_data),
                'max_area': max(board.get('width_mm', 0) * board.get('height_mm', 0) for board in input_data.boards_data),
                'avg_area': np.mean([board.get('width_mm', 0) * board.get('height_mm', 0) for board in input_data.boards_data])
            }
        }
        
        return {
            'parts_summary': parts_summary,
            'boards_summary': boards_summary,
            'material_type': input_data.material_type,
            'thickness_mm': input_data.thickness_mm,
            'complexity_score': input_data.complexity_score,
            # Nesting functionality removed for deployment
        }
    
    async def _analyze_with_openai(self, analysis_data: Dict[str, Any]) -> ScrapCalculationResult:
        """Analyze using OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Create prompt for OpenAI
        prompt = self._create_openai_prompt(analysis_data)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'gpt-4',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert manufacturing engineer specializing in material optimization and scrap factor calculation for laser cutting operations.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.3
            }
            
            async with self.session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_openai_response(result, analysis_data)
                else:
                    raise Exception(f"OpenAI API error: {response.status}")
                    
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to rule-based analysis
            return await self._analyze_with_rules(analysis_data)
    
    def _create_openai_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Create prompt for OpenAI analysis"""
        parts = analysis_data['parts_summary']
        boards = analysis_data['boards_summary']
        
        prompt = f"""
Analyze the following manufacturing data and provide an optimal scrap factor for laser cutting:

PARTS DATA:
- Total parts: {parts['total_parts']}
- Unique shapes: {parts['unique_shapes']}
- Size range: {parts['size_range']['min_area']:.0f} - {parts['size_range']['max_area']:.0f} mm²
- Average part size: {parts['size_range']['avg_area']:.0f} mm²
- Quantity range: {parts['quantity_distribution']['min_qty']} - {parts['quantity_distribution']['max_qty']}
- Average quantity: {parts['quantity_distribution']['avg_qty']:.1f}

BOARDS DATA:
- Available boards: {boards['total_boards']}
- Board size range: {boards['size_range']['min_area']:.0f} - {boards['size_range']['max_area']:.0f} mm²
- Average board size: {boards['size_range']['avg_area']:.0f} mm²

MATERIAL: {analysis_data['material_type']}, {analysis_data['thickness_mm']}mm thickness
COMPLEXITY SCORE: {analysis_data['complexity_score']:.2f}

Provide your response in JSON format:
{{
    "scrap_factor": <float between 1.0 and 2.0>,
    "confidence_score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation>",
    "estimated_utilization": <float between 0.0 and 1.0>,
    "optimization_suggestions": ["<suggestion1>", "<suggestion2>"]
}}
"""
        return prompt
    
    def _parse_openai_response(self, response: Dict[str, Any], analysis_data: Dict[str, Any]) -> ScrapCalculationResult:
        """Parse OpenAI API response"""
        try:
            content = response['choices'][0]['message']['content']
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                ai_result = json.loads(json_str)
                
                return ScrapCalculationResult(
                    scrap_factor=float(ai_result.get('scrap_factor', 1.2)),
                    confidence_score=float(ai_result.get('confidence_score', 0.8)),
                    reasoning=ai_result.get('reasoning', 'AI analysis completed'),
                    calculation_time=0.0,
                    optimization_suggestions=ai_result.get('optimization_suggestions', []),
                    estimated_utilization=float(ai_result.get('estimated_utilization', 0.8))
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing OpenAI response: {e}")
            # Fallback to rule-based calculation
            return self._calculate_rule_based_scrap(analysis_data)
    
    async def _analyze_with_rules(self, analysis_data: Dict[str, Any]) -> ScrapCalculationResult:
        """Analyze using rule-based system"""
        return self._calculate_rule_based_scrap(analysis_data)
    
    def _calculate_rule_based_scrap(self, analysis_data: Dict[str, Any]) -> ScrapCalculationResult:
        """Calculate scrap factor using rule-based system"""
        parts = analysis_data['parts_summary']
        complexity = analysis_data['complexity_score']
        
        # Base scrap factor
        base_scrap = 1.15
        
        # Adjustments based on complexity
        if complexity > 0.7:
            base_scrap += 0.15  # High complexity
        elif complexity > 0.4:
            base_scrap += 0.08  # Medium complexity
        else:
            base_scrap += 0.03  # Low complexity
        
        # Adjustments based on part count
        if parts['total_parts'] > 100:
            base_scrap += 0.05  # Many parts
        elif parts['total_parts'] > 50:
            base_scrap += 0.03  # Medium parts
        else:
            base_scrap += 0.01  # Few parts
        
        # Adjustments based on size variance
        size_variance = parts['size_range']['max_area'] / parts['size_range']['min_area'] if parts['size_range']['min_area'] > 0 else 1
        if size_variance > 10:
            base_scrap += 0.08  # High size variance
        elif size_variance > 5:
            base_scrap += 0.04  # Medium size variance
        
        # Material adjustments
        material_adjustments = {
            'steel': 0.0,
            'aluminum': 0.02,
            'stainless': 0.03,
            'copper': 0.05
        }
        base_scrap += material_adjustments.get(analysis_data['material_type'], 0.0)
        
        # Ensure reasonable bounds
        scrap_factor = max(1.05, min(1.8, base_scrap))
        
        # Calculate confidence based on data quality
        confidence = 0.8
        if parts['total_parts'] > 20:
            confidence += 0.1
        if parts['unique_shapes'] > 5:
            confidence += 0.05
        
        confidence = min(0.95, confidence)
        
        # Generate suggestions
        suggestions = []
        if complexity > 0.6:
            suggestions.append("Consider grouping similar parts for better utilization")
        if parts['total_parts'] > 100:
            suggestions.append("Large part count - consider batch processing")
        if size_variance > 8:
            suggestions.append("High size variance - consider separate processing for large/small parts")
        
        return ScrapCalculationResult(
            scrap_factor=scrap_factor,
            confidence_score=confidence,
            reasoning=f"Rule-based calculation: complexity={complexity:.2f}, parts={parts['total_parts']}, variance={size_variance:.1f}",
            calculation_time=0.0,
            optimization_suggestions=suggestions,
            estimated_utilization=1.0 / scrap_factor
        )
    
    async def _fallback_calculation(self, input_data: ScrapCalculationInput, start_time: float) -> ScrapCalculationResult:
        """Fallback calculation when AI fails"""
        return ScrapCalculationResult(
            scrap_factor=1.2,
            confidence_score=0.5,
            reasoning="Fallback calculation - AI analysis failed",
            calculation_time=time.time() - start_time,
            optimization_suggestions=["Consider manual scrap factor adjustment"],
            estimated_utilization=0.83
        )
    
    def clear_cache(self):
        """Clear the calculation cache"""
        with self.cache_lock:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'cache_size': len(self.cache),
                'cache_keys': list(self.cache.keys())[:10]  # First 10 keys
            }

# Global instance
ai_scrap_calculator = AIScrapCalculator()

async def calculate_scrap_factor_ai(parts_data: List[Dict[str, Any]], 
                                  boards_data: List[Dict[str, Any]], 
                                  material_type: str = "steel",
                                  thickness_mm: float = 1.0,
                                  openai_api_key: Optional[str] = None) -> ScrapCalculationResult:
    """Convenience function for AI scrap calculation"""
    if openai_api_key:
        ai_scrap_calculator.api_key = openai_api_key
    
    input_data = ScrapCalculationInput(
        parts_data=parts_data,
        boards_data=boards_data,
        material_type=material_type,
        thickness_mm=thickness_mm
    )
    
    # Calculate complexity score
    input_data.complexity_score = ai_scrap_calculator._calculate_complexity_score(parts_data)
    
    return await ai_scrap_calculator.calculate_scrap_factor_ai(input_data)
