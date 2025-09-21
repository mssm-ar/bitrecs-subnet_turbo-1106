import json
import re
import time
import bittensor as bt
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

class Gender(Enum):
    WOMEN = "W"
    MEN = "M"
    UNISEX = "U"

class Category(Enum):
    SHORTS = "SH"
    JACKETS = "J"
    PANTS = "P"
    TANKS = "T"
    HOODIES = "H"
    SHIRTS = "S"
    GEAR = "G"
    BAGS = "B"
    WATCHES = "WG"
    FITNESS_EQUIPMENT = "UG"

class Activity(Enum):
    RUNNING = "running"
    YOGA = "yoga"
    FITNESS = "fitness"
    GYM = "gym"
    WORKOUT = "workout"
    TRAINING = "training"

@dataclass
class ProductFeatures:
    sku: str
    name: str
    price: float
    gender: Gender
    category: Category
    activity: Optional[Activity]
    subcategory: str
    brand_collection: str
    price_tier: str
    eco_friendly: bool
    performance_fabric: bool
    sale_item: bool
    erin_recommends: bool

class ContextPreSelector:
    """
    Smart context pre-selector optimized for consensus participation and speed.
    Targets 22 products for 700 tokens with balanced relevance and diversity.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        
    def pre_select_context(self, query_sku: str, full_context: str, max_products: int = 22, num_recs: int = 5) -> str:
        """
        Main pre-selection function optimized for direct product selection.
        Selects exactly num_recs products with smart distribution.
        
        Args:
            query_sku: The SKU being queried (e.g., "WSH04")
            full_context: Full product catalog as JSON string
            max_products: Maximum number of products to return (default: 22 for 700 tokens)
            num_recs: Number of products to return (default: 5)
            
        Returns:
            JSON string of pre-selected products (exactly num_recs)
        """
        try:
            # Check cache first
            cache_key = f"{query_sku}_{hash(full_context)}_{num_recs}"
            if self._is_cached(cache_key):
                bt.logging.info(f"🎯 CACHE HIT: Using pre-selected context for {query_sku}")
                return self.cache[cache_key]
            
            # Parse products
            products = json.loads(full_context)
            if len(products) < num_recs:  # Don't pre-select if not enough products
                return full_context
                
            # Find query product
            query_product = self._find_product_by_sku(products, query_sku)
            if not query_product:
                bt.logging.warning(f"Query product {query_sku} not found in context")
                return full_context
                
            # Extract features from query product
            query_features = self._extract_product_features(query_product)
            
            # Direct selection based on query type
            selected_products = self._direct_product_selection(products, query_features, query_sku, num_recs)
            
            # Cache result
            result = json.dumps(selected_products)
            self._cache_result(cache_key, result)
            
            bt.logging.info(f"✅ DIRECT SELECTION: {len(products)} -> {len(selected_products)} products for {query_sku}")
            return result
            
        except Exception as e:
            bt.logging.error(f"Pre-selection failed: {e}")
            return full_context
    
    def _find_product_by_sku(self, products: List[Dict], sku: str) -> Optional[Dict]:
        """Find product by SKU (case-insensitive)"""
        for product in products:
            if product.get('sku', '').upper() == sku.upper():
                return product
        return None
    
    def _direct_product_selection(self, products: List[Dict], query_features: ProductFeatures, query_sku: str, num_recs: int) -> List[Dict]:
        """
        Direct product selection with smart distribution based on query type.
        For gendered products: 50% same gender, 20% same category, remaining for same season/related
        For non-gendered products: select same category products from context
        """
        # Check if query is gendered product
        is_gendered = query_features.gender != Gender.UNISEX
        
        if is_gendered:
            return self._select_gendered_products(products, query_features, query_sku, num_recs)
        else:
            return self._select_ungendered_products(products, query_features, query_sku, num_recs)
    
    def _select_gendered_products(self, products: List[Dict], query_features: ProductFeatures, query_sku: str, num_recs: int) -> List[Dict]:
        """Select products for gendered queries with 50% same gender, 20% same category, remaining for same season/related"""
        selected = []
        selected_skus = set()
        
        # Calculate flexible distribution numbers
        same_gender_target = max(1, int(num_recs * 0.5))  # 50% same gender
        same_category_target = max(1, int(num_recs * 0.2))  # 20% same category
        remaining_target = num_recs - same_gender_target - same_category_target  # Remaining for same season/related
        
        bt.logging.info(f"🎯 GENDERED SELECTION: {same_gender_target} same-gender, {same_category_target} same-category, {remaining_target} same-season/related")
        
        # Phase 1: Same gender products (50%)
        same_gender_count = 0
        for product in products:
            if same_gender_count >= same_gender_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                if candidate_features.gender == query_features.gender:
                    selected.append(product)
                    selected_skus.add(sku)
                    same_gender_count += 1
                    bt.logging.info(f"➕ Added same-gender product: {sku} ({candidate_features.gender})")
        
        # Phase 2: Same category products (20%)
        same_category_count = 0
        for product in products:
            if same_category_count >= same_category_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                if candidate_features.category == query_features.category:
                    selected.append(product)
                    selected_skus.add(sku)
                    same_category_count += 1
                    bt.logging.info(f"➕ Added same-category product: {sku} ({candidate_features.category})")
        
        # Phase 3: Same season or related products (remaining)
        remaining_count = 0
        for product in products:
            if remaining_count >= remaining_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                # Check for same season or smart complementary
                if (self._is_same_season(query_features, candidate_features) or 
                    self._is_smart_complementary(query_features, candidate_features)):
                    selected.append(product)
                    selected_skus.add(sku)
                    remaining_count += 1
                    bt.logging.info(f"➕ Added same-season/related product: {sku}")
        
        # If we still need more products, add any available
        if len(selected) < num_recs:
            bt.logging.info(f"🔧 FILLING REMAINING: Need {num_recs - len(selected)} more products")
            for product in products:
                if len(selected) >= num_recs:
                    break
                sku = product.get('sku', '')
                if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                    selected.append(product)
                    selected_skus.add(sku)
                    bt.logging.info(f"➕ Added filler product: {sku}")
        
        # Log final distribution
        if selected:
            total = len(selected)
            same_gender_pct = same_gender_count/total*100
            same_category_pct = same_category_count/total*100
            remaining_pct = remaining_count/total*100
            
            bt.logging.info(f"✅ GENDERED DISTRIBUTION: {same_gender_count}/{total} same-gender ({same_gender_pct:.1f}%), {same_category_count}/{total} same-category ({same_category_pct:.1f}%), {remaining_count}/{total} same-season/related ({remaining_pct:.1f}%)")
        
        return selected[:num_recs]  # Ensure we return exactly num_recs
    
    def _select_ungendered_products(self, products: List[Dict], query_features: ProductFeatures, query_sku: str, num_recs: int) -> List[Dict]:
        """Select products for non-gendered queries - focus on same category products"""
        selected = []
        selected_skus = set()
        
        bt.logging.info(f"🎯 UNGENDERED SELECTION: Selecting {num_recs} same-category products")
        
        # Phase 1: Same category products (primary focus)
        same_category_count = 0
        for product in products:
            if len(selected) >= num_recs:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                if candidate_features.category == query_features.category:
                    selected.append(product)
                    selected_skus.add(sku)
                    same_category_count += 1
                    bt.logging.info(f"➕ Added same-category product: {sku} ({candidate_features.category})")
        
        # Phase 2: If not enough same-category, add related products
        if len(selected) < num_recs:
            bt.logging.info(f"🔧 ADDING RELATED: Need {num_recs - len(selected)} more products")
            for product in products:
                if len(selected) >= num_recs:
                    break
                sku = product.get('sku', '')
                if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                    candidate_features = self._extract_product_features(product)
                    # Add unisex products or smart complementary
                    if (candidate_features.gender == Gender.UNISEX or 
                        self._is_smart_complementary(query_features, candidate_features)):
                        selected.append(product)
                        selected_skus.add(sku)
                        bt.logging.info(f"➕ Added related product: {sku}")
        
        # Phase 3: If still not enough, add any available
        if len(selected) < num_recs:
            bt.logging.info(f"🔧 FILLING REMAINING: Need {num_recs - len(selected)} more products")
            for product in products:
                if len(selected) >= num_recs:
                    break
                sku = product.get('sku', '')
                if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                    selected.append(product)
                    selected_skus.add(sku)
                    bt.logging.info(f"➕ Added filler product: {sku}")
        
        # Log final distribution
        if selected:
            total = len(selected)
            same_category_pct = same_category_count/total*100
            bt.logging.info(f"✅ UNGENDERED DISTRIBUTION: {same_category_count}/{total} same-category ({same_category_pct:.1f}%), {total - same_category_count}/{total} related/filler ({(total - same_category_count)/total*100:.1f}%)")
        
        return selected[:num_recs]  # Ensure we return exactly num_recs
    
    def _extract_product_features(self, product: Dict) -> ProductFeatures:
        """Extract comprehensive features from a product"""
        sku = product.get('sku', '')
        name = product.get('name', '')
        price = float(product.get('price', 0))
        
        # Extract gender from SKU prefix
        gender = self._extract_gender_from_sku(sku)
        
        # Extract category from SKU
        category = self._extract_category_from_sku(sku)
        
        # Extract activity from name
        activity = self._extract_activity_from_name(name)
        
        # Extract subcategory from name
        subcategory = self._extract_subcategory_from_name(name)
        
        # Extract brand/collection
        brand_collection = self._extract_brand_collection(name)
        
        # Determine price tier
        price_tier = self._determine_price_tier(price)
        
        # Extract special features
        eco_friendly = 'eco friendly' in name.lower()
        performance_fabric = 'performance fabric' in name.lower()
        sale_item = 'sale' in name.lower()
        erin_recommends = 'erin recommends' in name.lower()
        
        return ProductFeatures(
            sku=sku,
            name=name,
            price=price,
            gender=gender,
            category=category,
            activity=activity,
            subcategory=subcategory,
            brand_collection=brand_collection,
            price_tier=price_tier,
            eco_friendly=eco_friendly,
            performance_fabric=performance_fabric,
            sale_item=sale_item,
            erin_recommends=erin_recommends
        )
    
    def _extract_gender_from_sku(self, sku: str) -> Gender:
        """Extract gender from SKU prefix"""
        if sku.startswith('W'):
            return Gender.WOMEN
        elif sku.startswith('M'):
            return Gender.MEN
        else:
            return Gender.UNISEX
    
    def _extract_category_from_sku(self, sku: str) -> Category:
        """Extract category from SKU pattern"""
        if 'SH' in sku:
            return Category.SHORTS
        elif 'J' in sku and not sku.startswith('24-'):
            return Category.JACKETS
        elif 'P' in sku and not sku.startswith('24-'):
            return Category.PANTS
        elif 'T' in sku and not sku.startswith('24-'):
            return Category.TANKS
        elif 'H' in sku and not sku.startswith('24-'):
            return Category.HOODIES
        elif 'S' in sku and not sku.startswith('24-'):
            return Category.SHIRTS
        elif sku.startswith('24-WG') or sku.startswith('24-MG'):
            return Category.WATCHES
        elif sku.startswith('24-UG'):
            return Category.FITNESS_EQUIPMENT
        elif sku.startswith('24-WB') or sku.startswith('24-MB') or sku.startswith('24-UB'):
            return Category.BAGS
        else:
            return Category.GEAR
    
    def _extract_activity_from_name(self, name: str) -> Optional[Activity]:
        """Extract activity type from product name"""
        name_lower = name.lower()
        if 'running' in name_lower:
            return Activity.RUNNING
        elif 'yoga' in name_lower:
            return Activity.YOGA
        elif 'fitness' in name_lower:
            return Activity.FITNESS
        elif 'gym' in name_lower:
            return Activity.GYM
        elif 'workout' in name_lower:
            return Activity.WORKOUT
        elif 'training' in name_lower:
            return Activity.TRAINING
        return None
    
    def _extract_subcategory_from_name(self, name: str) -> str:
        """Extract subcategory from product name"""
        subcategories = [
            'compression', 'drawstring', 'bike', 'capri', 'leggings', 
            'tights', 'crew-neck', 'v-neck', 'tank', 'hoodie', 'sweatshirt',
            'jacket', 'pullover', 'zip', 'full-zip', 'half-zip'
        ]
        
        name_lower = name.lower()
        for sub in subcategories:
            if sub in name_lower:
                return sub
        return 'standard'
    
    def _extract_brand_collection(self, name: str) -> str:
        """Extract brand or collection from product name"""
        collections = [
            'new luma yoga collection', 'erin recommends', 'eco friendly',
            'performance fabrics', 'women sale', 'men sale'
        ]
        
        name_lower = name.lower()
        for collection in collections:
            if collection in name_lower:
                return collection
        return 'standard'
    
    def _determine_price_tier(self, price: float) -> str:
        """Determine price tier"""
        if price < 20:
            return 'budget'
        elif price < 40:
            return 'mid'
        elif price < 70:
            return 'premium'
        else:
            return 'luxury'
    
    def _collect_union_groups(self, products: List[Dict], query_features: ProductFeatures, query_sku: str) -> List[Dict]:
        """Collect products using OPTIMIZED UNION logic with MAXIMUM gender prioritization"""
        union_pool = set()
        
        # Check if query is gender-specific (not unisex)
        is_gender_specific = query_features.gender != Gender.UNISEX
        
        for product in products:
            if product.get('sku', '').upper() == query_sku.upper():
                continue  # Skip query product
                
            candidate_features = self._extract_product_features(product)
            
            # PRIORITY 1: Same gender (MAXIMUM weight for 70-80% target)
            if candidate_features.gender == query_features.gender:
                union_pool.add(json.dumps(product))
                # Triple weight for same gender to ensure 70-80% selection
                if is_gender_specific:
                    union_pool.add(json.dumps(product))
                    union_pool.add(json.dumps(product))
            
            # PRIORITY 2: Exact category match (high priority)
            if candidate_features.category == query_features.category:
                union_pool.add(json.dumps(product))
            
            # PRIORITY 3: Unisex products (medium priority)
            if candidate_features.gender == Gender.UNISEX:
                union_pool.add(json.dumps(product))
            
            # PRIORITY 4: Cross-gender for certain categories (minimal priority)
            if self._should_include_cross_gender(query_features, candidate_features):
                union_pool.add(json.dumps(product))
            
            # PRIORITY 5: Smart complementary logic (activity, collection, style-based)
            if self._is_smart_complementary(query_features, candidate_features):
                union_pool.add(json.dumps(product))
        
        # Convert back to product dictionaries
        return [json.loads(product_str) for product_str in union_pool]
    
    
    def _is_smart_complementary(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> bool:
        """OPTIMIZED complementary logic focusing on key relationships"""
        query_name = query_features.name.lower()
        candidate_name = candidate_features.name.lower()
        
        # Key activity-based complementary logic (simplified)
        activity_keywords = ['yoga', 'running', 'fitness', 'workout', 'training', 'gym']
        for activity in activity_keywords:
            if activity in query_name and activity in candidate_name:
                return True
        
        # Key collection-based complementary logic
        collections = ['new luma yoga', 'erin recommends', 'eco friendly', 'performance fabrics']
        for collection in collections:
            if collection in query_name and collection in candidate_name:
                return True
        
        # Key style-based complementary logic (simplified)
        style_pairs = [
            ('tee', 'jacket'), ('tank', 'hoodie'), ('short', 'pant'),
            ('legging', 'tank'), ('tights', 'tee')
        ]
        for style1, style2 in style_pairs:
            if (style1 in query_name and style2 in candidate_name) or \
               (style2 in query_name and style1 in candidate_name):
                return True
        
        return False
    
    def _should_include_cross_gender(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> bool:
        """OPTIMIZED cross-gender inclusion for minimal variety"""
        # Only include cross-gender for gender-neutral categories
        cross_gender_categories = [
            Category.GEAR, Category.BAGS, Category.WATCHES, Category.FITNESS_EQUIPMENT
        ]
        
        return candidate_features.category in cross_gender_categories
    
    def _is_same_season(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> bool:
        """Determine if products are from the same season based on name/collection patterns"""
        query_name = query_features.name.lower()
        candidate_name = candidate_features.name.lower()
        
        # Season indicators in product names
        spring_summer_indicators = [
            'spring', 'summer', 'tank', 'short', 'sleeveless', 'lightweight', 
            'breathable', 'cool', 'airy', 'bikini', 'swim', 'beach'
        ]
        
        fall_winter_indicators = [
            'fall', 'winter', 'warm', 'thermal', 'fleece', 'hoodie', 'sweatshirt',
            'jacket', 'coat', 'long sleeve', 'heavy', 'insulated', 'wool'
        ]
        
        # Check if both products have similar season indicators
        query_spring_summer = any(indicator in query_name for indicator in spring_summer_indicators)
        query_fall_winter = any(indicator in query_name for indicator in fall_winter_indicators)
        
        candidate_spring_summer = any(indicator in candidate_name for indicator in spring_summer_indicators)
        candidate_fall_winter = any(indicator in candidate_name for indicator in fall_winter_indicators)
        
        # Same season if both are spring/summer or both are fall/winter
        if query_spring_summer and candidate_spring_summer:
            return True
        if query_fall_winter and candidate_fall_winter:
            return True
        
        # Also check collection-based season matching
        query_collection = query_features.brand_collection.lower()
        candidate_collection = candidate_features.brand_collection.lower()
        
        # If both are from the same collection, they're likely same season
        if query_collection == candidate_collection and query_collection != 'standard':
            return True
        
        # Check for seasonal activity patterns
        if query_features.activity and candidate_features.activity:
            # Running and fitness are more spring/summer
            spring_activities = [Activity.RUNNING, Activity.FITNESS]
            # Yoga and gym can be year-round but lean towards fall/winter
            fall_activities = [Activity.YOGA, Activity.GYM]
            
            if (query_features.activity in spring_activities and candidate_features.activity in spring_activities) or \
               (query_features.activity in fall_activities and candidate_features.activity in fall_activities):
                return True
        
        return False
    
    def _score_for_consensus(self, products: List[Dict], query_features: ProductFeatures, query_sku: str) -> List[Tuple[Dict, float]]:
        """Score products for consensus optimization"""
        scored_products = []
        
        for product in products:
            candidate_features = self._extract_product_features(product)
            score = self._calculate_consensus_score(query_features, candidate_features)
            scored_products.append((product, score))
        
        # Sort by score (highest first)
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return scored_products
    
    def _calculate_consensus_score(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> float:
        """
        Calculate score optimized for consensus participation with MAXIMUM gender weighting.
        Gives 75%+ weight to gender matching for 70-80% same-gender selection.
        """
        score = 0.0
        
        # MAXIMUM gender weighting (75%+ for 70-80% same-gender selection)
        if query_features.gender == candidate_features.gender:
            score += 0.6  # Same gender (increased to 75% for 70-80% target)
        elif candidate_features.gender == Gender.UNISEX:
            score += 0.15  # Unisex products (reduced to prioritize same gender)
        elif self._should_include_cross_gender(query_features, candidate_features):
            score += 0.05  # Cross-gender for minimal variety (reduced further)
        
        # Reduced other weights to maintain gender dominance
        if query_features.category == candidate_features.category:
            score += 0.3  # Category match (reduced to prioritize gender)
        
        if self._name_similarity(query_features.name, candidate_features.name) > 0.5:
            score += 0.05  # Name/attribute analysis (minimal weight)
        
        if self._price_similarity(query_features.price, candidate_features.price) > 0.7:
            score += 0.03  # Price similarity (minimal weight)
        
        # Smart complementary bonus (activity, collection, style-based)
        if self._is_smart_complementary(query_features, candidate_features):
            score += 0.05  # Smart complementary bonus (reduced)
        
        return score
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity score"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def _price_similarity(self, price1: float, price2: float) -> float:
        """Calculate price similarity score (0-1)"""
        if price1 == 0 or price2 == 0:
            return 0.0
        
        diff = abs(price1 - price2) / max(price1, price2)
        
        if diff <= 0.1:  # Within 10%
            return 1.0
        elif diff <= 0.2:  # Within 20%
            return 0.8
        elif diff <= 0.3:  # Within 30%
            return 0.6
        elif diff <= 0.5:  # Within 50%
            return 0.4
        else:
            return 0.2
    
    def _balanced_selection(self, scored_products: List[Tuple[Dict, float]], max_products: int, query_sku: str = None) -> List[Dict]:
        """STRICT balanced selection with specific distribution: 50% same gender, 20% same category, 20% same season, 10% others"""
        if not scored_products:
            return []
        
        # Remove duplicates from scored products first
        seen_skus = set()
        unique_scored_products = []
        for product, score in scored_products:
            sku = product.get('sku', '')
            if sku and sku not in seen_skus:
                unique_scored_products.append((product, score))
                seen_skus.add(sku)
        
        scored_products = unique_scored_products
        
        # Get query product features for comparison
        query_features = None
        for product, score in scored_products:
            if product.get('sku', '').upper() == query_sku.upper():
                query_features = self._extract_product_features(product)
                break
        
        if not query_features:
            # Fallback: just return top products
            return [product for product, score in scored_products[:max_products]]
        
        # Categorize products based on the new distribution logic
        same_gender = []
        same_category = []
        same_season = []
        others = []
        
        for product, score in scored_products:
            sku = product.get('sku', '')
            if sku.upper() == query_sku.upper():
                continue  # Skip query product
                
            product_features = self._extract_product_features(product)
            
            # 50% same gender products
            if product_features.gender == query_features.gender:
                same_gender.append((product, score))
            
            # 20% same category products (excluding same gender to avoid double counting)
            elif product_features.category == query_features.category:
                same_category.append((product, score))
            
            # 20% same season products (based on product name/collection)
            elif self._is_same_season(query_features, product_features):
                same_season.append((product, score))
            
            # 10% others (related to query)
            else:
                others.append((product, score))
        
        # Calculate flexible distribution numbers
        total_needed = min(max_products, len(scored_products))
        
        # Flexible numbers within each parameter
        same_gender_target = max(1, int(total_needed * 0.5))  # 50% ± flexibility
        same_category_target = max(1, int(total_needed * 0.2))  # 20% ± flexibility  
        same_season_target = max(1, int(total_needed * 0.2))  # 20% ± flexibility
        others_target = max(1, int(total_needed * 0.1))  # 10% ± flexibility
        
        # Ensure we have at least num_recs products
        if total_needed < max_products:
            # Distribute remaining slots proportionally
            remaining = max_products - total_needed
            same_gender_target += int(remaining * 0.5)
            same_category_target += int(remaining * 0.2)
            same_season_target += int(remaining * 0.2)
            others_target += int(remaining * 0.1)
        
        selected = []
        selected_skus = set()
        
        # Select products in priority order
        bt.logging.info(f"🎯 Distribution targets: {same_gender_target} same-gender, {same_category_target} same-category, {same_season_target} same-season, {others_target} others")
        
        # 1. Same gender products (50%)
        same_gender_selected = 0
        for product, score in same_gender[:same_gender_target]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus:
                selected.append(product)
                selected_skus.add(sku)
                same_gender_selected += 1
        
        # 2. Same category products (20%)
        same_category_selected = 0
        for product, score in same_category[:same_category_target]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus:
                selected.append(product)
                selected_skus.add(sku)
                same_category_selected += 1
        
        # 3. Same season products (20%)
        same_season_selected = 0
        for product, score in same_season[:same_season_target]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus:
                selected.append(product)
                selected_skus.add(sku)
                same_season_selected += 1
        
        # 4. Others (10%)
        others_selected = 0
        for product, score in others[:others_target]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus:
                selected.append(product)
                selected_skus.add(sku)
                others_selected += 1
        
        # Log distribution for debugging
        if selected:
            total = len(selected)
            same_gender_pct = same_gender_selected/total*100
            same_category_pct = same_category_selected/total*100
            same_season_pct = same_season_selected/total*100
            others_pct = others_selected/total*100
            
            bt.logging.info(f"✅ Distribution: {same_gender_selected}/{total} same-gender ({same_gender_pct:.1f}%), {same_category_selected}/{total} same-category ({same_category_pct:.1f}%), {same_season_selected}/{total} same-season ({same_season_pct:.1f}%), {others_selected}/{total} others ({others_pct:.1f}%)")
        
        return selected
    
    def _ensure_minimum_products(self, selected_products: List[Dict], all_products: List[Dict], query_sku: str, num_recs: int) -> List[Dict]:
        """STRICT method to ensure we have at least num_recs UNIQUE products following distribution logic"""
        # Remove any duplicates from current selection
        seen_skus = set()
        unique_products = []
        for product in selected_products:
            sku = product.get('sku', '')
            if sku and sku not in seen_skus:
                unique_products.append(product)
                seen_skus.add(sku)
        
        selected_products = unique_products
        selected_skus = seen_skus
        original_count = len(selected_products)
        
        # Get query product features for distribution logic
        query_product = self._find_product_by_sku(all_products, query_sku)
        query_features = None
        if query_product:
            query_features = self._extract_product_features(query_product)
        
        bt.logging.warning(f"🔧 ENSURING MINIMUM: Need {num_recs} UNIQUE products, have {original_count}")
        
        if not query_features:
            # Fallback: just add any products
            for product in all_products:
                if len(selected_products) >= num_recs:
                    break
                sku = product.get('sku', '')
                if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                    selected_products.append(product)
                    selected_skus.add(sku)
            return selected_products
        
        # Calculate distribution targets for remaining products needed
        remaining_needed = num_recs - len(selected_products)
        if remaining_needed <= 0:
            return selected_products
        
        # Calculate targets for distribution
        same_gender_target = max(1, int(remaining_needed * 0.5))
        same_category_target = max(1, int(remaining_needed * 0.2))
        same_season_target = max(1, int(remaining_needed * 0.2))
        others_target = max(1, int(remaining_needed * 0.1))
        
        bt.logging.info(f"🎯 Adding {remaining_needed} more products: {same_gender_target} same-gender, {same_category_target} same-category, {same_season_target} same-season, {others_target} others")
        
        # Strategy 1: Add same gender products (50% priority)
        same_gender_added = 0
        for product in all_products:
            if len(selected_products) >= num_recs or same_gender_added >= same_gender_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                if candidate_features.gender == query_features.gender:
                    selected_products.append(product)
                    selected_skus.add(sku)
                    same_gender_added += 1
                    bt.logging.info(f"➕ Added same-gender product: {sku} ({candidate_features.gender})")
        
        # Strategy 2: Add same category products (20% priority)
        same_category_added = 0
        for product in all_products:
            if len(selected_products) >= num_recs or same_category_added >= same_category_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                if candidate_features.category == query_features.category:
                    selected_products.append(product)
                    selected_skus.add(sku)
                    same_category_added += 1
                    bt.logging.info(f"➕ Added same-category product: {sku} ({candidate_features.category})")
        
        # Strategy 3: Add same season products (20% priority)
        same_season_added = 0
        for product in all_products:
            if len(selected_products) >= num_recs or same_season_added >= same_season_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                candidate_features = self._extract_product_features(product)
                if self._is_same_season(query_features, candidate_features):
                    selected_products.append(product)
                    selected_skus.add(sku)
                    same_season_added += 1
                    bt.logging.info(f"➕ Added same-season product: {sku}")
        
        # Strategy 4: Add others (10% priority)
        others_added = 0
        for product in all_products:
            if len(selected_products) >= num_recs or others_added >= others_target:
                break
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                selected_products.append(product)
                selected_skus.add(sku)
                others_added += 1
                bt.logging.info(f"➕ Added other product: {sku}")
        
        # Strategy 5: If still not enough, add any remaining products
        if len(selected_products) < num_recs:
            bt.logging.warning(f"⚠️ Still need {num_recs - len(selected_products)} more products, adding any available...")
            for product in all_products:
                if len(selected_products) >= num_recs:
                    break
                sku = product.get('sku', '')
                if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                    selected_products.append(product)
                    selected_skus.add(sku)
                    bt.logging.info(f"➕ Added emergency product: {sku}")
        
        final_count = len(selected_products)
        final_unique_count = len(selected_skus)
        bt.logging.info(f"✅ ENSURED MINIMUM: {original_count} → {final_count} products ({final_unique_count} unique) (target: {num_recs})")
        
        if final_count < num_recs or final_unique_count != final_count:
            bt.logging.error(f"❌ CRITICAL: Still only have {final_count} products ({final_unique_count} unique), need {num_recs}")
        
        return selected_products
    
    def _get_similar_products_from_context(self, needed_count: int, query_sku: str, selected_skus: Set[str], query_category: Optional[Category]) -> List[Dict]:
        """Get similar and related products from context.json to ensure minimum product count"""
        try:
            import os
            # Get the project root directory (3 levels up from this file)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            context_file_path = os.path.join(project_root, 'context.json')
            
            with open(context_file_path, 'r') as f:
                context_products = json.load(f)
            
            # Filter out query SKU and already selected SKUs
            available_products = [
                product for product in context_products
                if (product.get('sku', '').upper() != query_sku.upper() and 
                    product.get('sku', '') not in selected_skus)
            ]
            
            if not available_products:
                bt.logging.warning("No available products in context.json for similar selection")
                return []
            
            # Score products by similarity to query
            scored_products = []
            for product in available_products:
                score = self._calculate_similarity_score(product, query_sku, query_category)
                scored_products.append((product, score))
            
            # Sort by similarity score (highest first)
            scored_products.sort(key=lambda x: x[1], reverse=True)
            
            # Select the most similar products
            similar_count = min(needed_count, len(scored_products))
            similar_products = [product for product, score in scored_products[:similar_count]]
            
            bt.logging.info(f"🎯 Selected {len(similar_products)} similar products from context.json")
            return similar_products
            
        except Exception as e:
            bt.logging.error(f"Failed to get similar products from context.json: {e}")
            return []
    
    def _calculate_similarity_score(self, product: Dict, query_sku: str, query_category: Optional[Category]) -> float:
        """Calculate similarity score between a product and the query"""
        try:
            product_features = self._extract_product_features(product)
            score = 0.0
            
            # High priority: Same category
            if query_category and product_features.category == query_category:
                score += 0.4
            
            # Medium priority: Same gender (if query is gender-specific)
            if query_sku.startswith(('W', 'M')) and not product.get('sku', '').startswith('24-'):
                if query_sku.startswith('W') and product.get('sku', '').startswith('W'):
                    score += 0.3
                elif query_sku.startswith('M') and product.get('sku', '').startswith('M'):
                    score += 0.3
            
            # Medium priority: Unisex products (good fallback)
            if product.get('sku', '').startswith('24-'):
                score += 0.2
            
            # Low priority: Similar price range
            try:
                query_price = float(query_sku.split('-')[1]) if '-' in query_sku else 50.0  # Fallback price
                product_price = float(product.get('price', 0))
                if product_price > 0:
                    price_diff = abs(query_price - product_price) / max(query_price, product_price)
                    if price_diff <= 0.3:  # Within 30%
                        score += 0.1
            except:
                pass
            
            # Low priority: Name similarity
            query_name_words = set(query_sku.lower().split())
            product_name_words = set(product.get('name', '').lower().split())
            if query_name_words and product_name_words:
                intersection = len(query_name_words.intersection(product_name_words))
                union = len(query_name_words.union(product_name_words))
                if union > 0:
                    score += 0.1 * (intersection / union)
            
            return score
            
        except Exception as e:
            bt.logging.debug(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if result is cached and not expired"""
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.cache_ttl:
                return True
            else:
                # Remove expired cache
                self.cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        return False
    
    def _cache_result(self, cache_key: str, result: str) -> None:
        """Cache the result with timestamp"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()


# Global instance for easy access
_preselector = ContextPreSelector()

def pre_select_context(query_sku: str, full_context: str, max_products: int = 22, num_recs: int = 5) -> str:
    """
    Convenience function for direct product selection from context.
    Returns exactly num_recs products with smart distribution.
    
    Args:
        query_sku: The SKU being queried (e.g., "WSH04")
        full_context: Full product catalog as JSON string
        max_products: Maximum number of products to return (default: 22 for 700 tokens)
        num_recs: Number of products to return (default: 5)
        
    Returns:
        JSON string of pre-selected products (exactly num_recs)
    """
    return _preselector.pre_select_context(query_sku, full_context, max_products, num_recs)
