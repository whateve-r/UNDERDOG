"""
Test Suite for Fuzzy Logic (Mamdani Inference)
Tests membership functions, rule evaluation, and confidence scoring.
"""
import pytest
import numpy as np

from underdog.strategies.fuzzy_logic.mamdani_inference import (
    TriangularMF, TrapezoidalMF, GaussianMF,
    FuzzyRule, MamdaniInference, ConfidenceScorer
)


class TestMembershipFunctions:
    """Test fuzzy membership functions"""
    
    def test_triangular_mf(self):
        """Test triangular membership function"""
        mf = TriangularMF(a=0.0, b=0.5, c=1.0)
        
        # Test at key points
        assert mf.evaluate(0.0) == 0.0  # Left edge
        assert mf.evaluate(0.5) == 1.0  # Peak
        assert mf.evaluate(1.0) == 0.0  # Right edge
        assert mf.evaluate(0.25) == pytest.approx(0.5, abs=0.01)
    
    def test_trapezoidal_mf(self):
        """Test trapezoidal membership function"""
        mf = TrapezoidalMF(a=0.0, b=0.3, c=0.7, d=1.0)
        
        # Test at key points
        assert mf.evaluate(0.0) == 0.0  # Left edge
        assert mf.evaluate(0.3) == 1.0  # Left plateau
        assert mf.evaluate(0.5) == 1.0  # Middle of plateau
        assert mf.evaluate(0.7) == 1.0  # Right plateau
        assert mf.evaluate(1.0) == 0.0  # Right edge
    
    def test_gaussian_mf(self):
        """Test Gaussian membership function"""
        mf = GaussianMF(mean=0.5, sigma=0.1)
        
        # Peak at mean
        assert mf.evaluate(0.5) == 1.0
        
        # Symmetric around mean
        assert mf.evaluate(0.4) == pytest.approx(mf.evaluate(0.6), abs=0.01)
        
        # Decreases away from mean
        assert mf.evaluate(0.3) < mf.evaluate(0.4)
    
    def test_mf_boundary_conditions(self):
        """Test membership functions at boundaries"""
        tri_mf = TriangularMF(a=0.0, b=0.5, c=1.0)
        
        # Values outside range should return 0
        assert tri_mf.evaluate(-0.5) == 0.0
        assert tri_mf.evaluate(1.5) == 0.0


class TestFuzzyRule:
    """Test fuzzy rule creation and evaluation"""
    
    def test_rule_creation(self):
        """Test creating a fuzzy rule"""
        rule = FuzzyRule(
            antecedents={'price_position': 'high', 'momentum': 'strong'},
            consequent='buy',
            strength=0.8
        )
        
        assert rule.antecedents['price_position'] == 'high'
        assert rule.consequent == 'buy'
        assert rule.strength == 0.8
    
    def test_rule_evaluation(self):
        """Test fuzzy rule evaluation with min operator"""
        rule = FuzzyRule(
            antecedents={'indicator_a': 'high', 'indicator_b': 'medium'},
            consequent='buy',
            strength=1.0
        )
        
        # Input memberships
        memberships = {
            'indicator_a': {'high': 0.8, 'medium': 0.2, 'low': 0.0},
            'indicator_b': {'high': 0.3, 'medium': 0.7, 'low': 0.0}
        }
        
        # Rule fires with min(0.8, 0.7) = 0.7
        activation = rule.evaluate(memberships)
        assert activation == 0.7


class TestMamdaniInference:
    """Test Mamdani inference engine"""
    
    def test_fuzzification(self):
        """Test fuzzification of crisp inputs"""
        # Create simple fuzzy system
        inference = MamdaniInference()
        
        # Add linguistic variables
        inference.add_variable('price', {
            'low': TriangularMF(0.0, 0.0, 0.5),
            'medium': TriangularMF(0.0, 0.5, 1.0),
            'high': TriangularMF(0.5, 1.0, 1.0)
        })
        
        # Fuzzify crisp value
        memberships = inference.fuzzify('price', 0.6)
        
        # Should have membership in 'medium' and 'high'
        assert 'medium' in memberships
        assert 'high' in memberships
        assert memberships['medium'] > 0
        assert memberships['high'] > 0
    
    def test_rule_inference(self):
        """Test inference with multiple rules"""
        inference = MamdaniInference()
        
        # Add variables
        inference.add_variable('rsi', {
            'oversold': TriangularMF(0.0, 0.0, 30.0),
            'neutral': TriangularMF(30.0, 50.0, 70.0),
            'overbought': TriangularMF(70.0, 100.0, 100.0)
        })
        
        inference.add_variable('confidence', {
            'low': TriangularMF(0.0, 0.0, 0.5),
            'high': TriangularMF(0.5, 1.0, 1.0)
        })
        
        # Add rules
        inference.add_rule(FuzzyRule(
            antecedents={'rsi': 'oversold'},
            consequent='high',
            strength=1.0
        ))
        
        inference.add_rule(FuzzyRule(
            antecedents={'rsi': 'overbought'},
            consequent='low',
            strength=1.0
        ))
        
        # Infer with RSI = 25 (oversold)
        result = inference.infer({'rsi': 25.0})
        
        # Should return high confidence
        assert result > 0.5
    
    def test_defuzzification_centroid(self):
        """Test centroid defuzzification method"""
        inference = MamdaniInference()
        
        # Create output fuzzy sets
        output_sets = {
            'low': TriangularMF(0.0, 0.0, 0.5),
            'high': TriangularMF(0.5, 1.0, 1.0)
        }
        
        # Activated output (high confidence)
        activated = {'high': 0.8, 'low': 0.2}
        
        crisp = inference.defuzzify(activated, output_sets, method='centroid')
        
        # Should be weighted toward 'high'
        assert crisp > 0.5


class TestConfidenceScorer:
    """Test confidence scoring system"""
    
    def test_confidence_scorer_initialization(self):
        """Test ConfidenceScorer initialization"""
        scorer = ConfidenceScorer()
        
        assert hasattr(scorer, 'inference_engine')
    
    def test_score_from_indicators(self):
        """Test scoring from technical indicators"""
        scorer = ConfidenceScorer()
        
        indicators = {
            'rsi': 35.0,  # Slightly oversold
            'bb_position': 0.2,  # Near lower band
            'cci': -150.0,  # Oversold
            'volume_ratio': 1.5  # Above average volume
        }
        
        confidence = scorer.score(indicators)
        
        # Should return confidence between 0 and 1
        assert 0.0 <= confidence <= 1.0
    
    def test_score_consistency(self):
        """Test that same inputs produce same scores"""
        scorer = ConfidenceScorer()
        
        indicators = {
            'rsi': 50.0,
            'bb_position': 0.5
        }
        
        score1 = scorer.score(indicators)
        score2 = scorer.score(indicators)
        
        assert score1 == score2
    
    def test_score_from_yaml_rules(self, tmp_path):
        """Test loading fuzzy rules from YAML"""
        # Create temporary YAML file
        yaml_content = """
variables:
  rsi:
    oversold: [0, 0, 30]
    neutral: [30, 50, 70]
    overbought: [70, 100, 100]
  
  confidence:
    low: [0, 0, 0.5]
    high: [0.5, 1.0, 1.0]

rules:
  - if:
      rsi: oversold
    then: high
    weight: 1.0
  
  - if:
      rsi: overbought
    then: low
    weight: 1.0
"""
        
        yaml_file = tmp_path / "fuzzy_rules.yaml"
        yaml_file.write_text(yaml_content)
        
        # Load from YAML
        scorer = ConfidenceScorer.from_yaml(str(yaml_file))
        
        # Test scoring
        confidence_oversold = scorer.score({'rsi': 25.0})
        confidence_overbought = scorer.score({'rsi': 85.0})
        
        # Oversold should give high confidence, overbought low
        assert confidence_oversold > 0.5
        assert confidence_overbought < 0.5


class TestFuzzyLogicEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_antecedents(self):
        """Test rule with empty antecedents"""
        rule = FuzzyRule(
            antecedents={},
            consequent='neutral',
            strength=0.5
        )
        
        # Should handle gracefully
        assert rule.consequent == 'neutral'
    
    def test_missing_membership_function(self):
        """Test with missing membership function"""
        inference = MamdaniInference()
        
        inference.add_variable('test', {
            'low': TriangularMF(0, 0, 0.5)
        })
        
        # Try to fuzzify with non-existent MF
        memberships = inference.fuzzify('test', 0.8)
        
        # Should return available memberships
        assert 'low' in memberships
    
    def test_invalid_input_range(self):
        """Test with input outside expected range"""
        mf = TriangularMF(0.0, 0.5, 1.0)
        
        # Far outside range
        assert mf.evaluate(10.0) == 0.0
        assert mf.evaluate(-10.0) == 0.0


class TestFuzzySystemIntegration:
    """Integration tests for complete fuzzy system"""
    
    def test_complete_fuzzy_workflow(self):
        """Test complete workflow: fuzzify → infer → defuzzify"""
        # Create inference engine
        inference = MamdaniInference()
        
        # Input variables
        inference.add_variable('price_to_ma', {
            'below': TriangularMF(-1.0, -1.0, 0.0),
            'at': TriangularMF(-0.1, 0.0, 0.1),
            'above': TriangularMF(0.0, 1.0, 1.0)
        })
        
        inference.add_variable('volume', {
            'low': TriangularMF(0.0, 0.0, 0.5),
            'high': TriangularMF(0.5, 1.0, 1.0)
        })
        
        # Output variable
        inference.add_variable('confidence', {
            'low': TriangularMF(0.0, 0.0, 0.5),
            'medium': TriangularMF(0.25, 0.5, 0.75),
            'high': TriangularMF(0.5, 1.0, 1.0)
        })
        
        # Rules
        inference.add_rule(FuzzyRule(
            antecedents={'price_to_ma': 'below', 'volume': 'high'},
            consequent='high',
            strength=1.0
        ))
        
        inference.add_rule(FuzzyRule(
            antecedents={'price_to_ma': 'above', 'volume': 'low'},
            consequent='low',
            strength=1.0
        ))
        
        # Test case 1: Price below MA with high volume (bullish)
        confidence_bullish = inference.infer({
            'price_to_ma': -0.05,
            'volume': 0.8
        })
        
        # Test case 2: Price above MA with low volume (bearish)
        confidence_bearish = inference.infer({
            'price_to_ma': 0.05,
            'volume': 0.3
        })
        
        # Bullish should have higher confidence
        assert confidence_bullish > confidence_bearish


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
