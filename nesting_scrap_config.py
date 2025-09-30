"""
Configuration management for nesting scrap and leftover analysis
"""

class NestingScrapConfig:
    """Configuration manager for leftover strip analysis"""
    
    def __init__(self):
        self.config = {
            'leftover_threshold_percent': 20.0,  # 20% minimum to qualify as leftover
            'min_leftover_width_mm': 180.0,
            'min_leftover_height_mm': 600.0,
            'scrap_margin_percent': 5.0,
            'enable_leftover_analysis': True,
            'cut_line_thickness': 4.0,
            'leftover_opacity': 0.3
        }
    
    def update_config(self, **kwargs):
        """Update configuration with provided values"""
        self.config.update(kwargs)
    
    def get_config(self):
        """Get current configuration"""
        return self.config.copy()


# Global config instance
nesting_scrap_config = NestingScrapConfig()
