# =============================================================================
# MARKET BASKET ANALYSIS - GROCERIES DATASET
# Step-by-Step Implementation for Jupyter Notebook
# =============================================================================

# Step 1: Import Required Libraries
# ---------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# Step 2: Load and Explore the Dataset
# ------------------------------------
# Load the groceries dataset
file_path = r"C:\Users\kervi\Documents\Data mining\Assignment\groceries.csv"

# Read the CSV file
df = pd.read_csv(file_path, header=None)

# Convert to transaction format
transactions = []
for index, row in df.iterrows():
    # Remove NaN values and empty strings
    transaction = [str(item).strip() for item in row.dropna() if str(item).strip() != 'nan' and str(item).strip() != '']
    if transaction:
        transactions.append(transaction)

print(f"📊 Dataset loaded successfully!")
print(f"📊 Total transactions: {len(transactions)}")
print(f"📊 Sample transactions:")
for i in range(5):
    print(f"   {i+1}: {transactions[i]}")

# Step 3: Basic Data Analysis
# ---------------------------
# Calculate basic statistics
transaction_lengths = [len(trans) for trans in transactions]
all_items = [item for trans in transactions for item in trans]
unique_items = set(all_items)

print(f"\n📈 DATASET STATISTICS:")
print(f"   - Total transactions: {len(transactions):,}")
print(f"   - Unique items: {len(unique_items):,}")
print(f"   - Total item purchases: {len(all_items):,}")
print(f"   - Average items per transaction: {np.mean(transaction_lengths):.2f}")
print(f"   - Min items per transaction: {min(transaction_lengths)}")
print(f"   - Max items per transaction: {max(transaction_lengths)}")

# Step 4: Find Most Frequent Items
# ---------------------------------
item_counts = Counter(all_items)
top_20_items = item_counts.most_common(20)

print(f"\n🏆 TOP 20 MOST FREQUENT ITEMS:")
print(f"{'Rank':<4} {'Item':<25} {'Frequency':<10} {'Support %':<10}")
print("-" * 55)

for rank, (item, count) in enumerate(top_20_items, 1):
    support = (count / len(transactions)) * 100
    print(f"{rank:<4} {item:<25} {count:<10} {support:<10.2f}")

# Step 5: Visualize Item Frequencies
# ----------------------------------
plt.figure(figsize=(15, 6))

# Top items frequency chart
plt.subplot(1, 2, 1)
items, counts = zip(*top_20_items[:15])
plt.barh(range(len(items)), counts, color='skyblue')
plt.yticks(range(len(items)), [item[:20] for item in items])
plt.xlabel('Frequency')
plt.title('Top 15 Most Frequent Items')
plt.gca().invert_yaxis()

# Transaction length distribution
plt.subplot(1, 2, 2)
plt.hist(transaction_lengths, bins=20, color='lightgreen', alpha=0.7)
plt.xlabel('Number of Items per Transaction')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Sizes')

plt.tight_layout()
plt.show()

# Step 6: Implement Apriori Algorithm Functions
# ----------------------------------------------

def calculate_support(itemset, transactions):
    """Calculate support for an itemset"""
    count = sum(1 for trans in transactions if set(itemset).issubset(set(trans)))
    return count / len(transactions)

def get_frequent_1_itemsets(transactions, min_support):
    """Get frequent 1-itemsets"""
    item_counts = Counter([item for trans in transactions for item in trans])
    frequent_items = []
    
    for item, count in item_counts.items():
        support = count / len(transactions)
        if support >= min_support:
            frequent_items.append(([item], support))
    
    return sorted(frequent_items, key=lambda x: x[1], reverse=True)

def generate_candidates(frequent_itemsets, k):
    """Generate candidate k-itemsets"""
    candidates = []
    n = len(frequent_itemsets)
    
    for i in range(n):
        for j in range(i + 1, n):
            itemset1 = frequent_itemsets[i][0]
            itemset2 = frequent_itemsets[j][0]
            
            # Merge itemsets
            candidate = sorted(list(set(itemset1 + itemset2)))
            if len(candidate) == k and candidate not in candidates:
                candidates.append(candidate)
    
    return candidates

def apriori_algorithm(transactions, min_support):
    """Complete Apriori algorithm implementation"""
    print(f"🚀 Running Apriori Algorithm (min_support = {min_support*100:.1f}%)")
    
    # Find frequent 1-itemsets
    frequent_itemsets = []
    frequent_1 = get_frequent_1_itemsets(transactions, min_support)
    frequent_itemsets.append(frequent_1)
    
    print(f"   ✅ Level 1: {len(frequent_1)} frequent itemsets")
    
    k = 2
    while True:
        # Generate candidates
        candidates = generate_candidates(frequent_itemsets[k-2], k)
        
        if not candidates:
            break
        
        # Calculate support for candidates
        frequent_k = []
        for candidate in candidates:
            support = calculate_support(candidate, transactions)
            if support >= min_support:
                frequent_k.append((candidate, support))
        
        if not frequent_k:
            break
        
        frequent_k = sorted(frequent_k, key=lambda x: x[1], reverse=True)
        frequent_itemsets.append(frequent_k)
        print(f"   ✅ Level {k}: {len(frequent_k)} frequent itemsets")
        
        k += 1
    
    print(f"✅ Apriori completed: Found {len(frequent_itemsets)} levels")
    return frequent_itemsets

# Step 7: Run Apriori Algorithm
# ------------------------------
# Set minimum support threshold (0.5% = items appearing in at least 0.5% of transactions)
min_support = 0.005

# Run the algorithm
frequent_itemsets = apriori_algorithm(transactions, min_support)

# Display results
print(f"\n📋 FREQUENT ITEMSETS SUMMARY:")
for level, itemsets in enumerate(frequent_itemsets, 1):
    print(f"   Level {level}: {len(itemsets)} frequent {level}-itemsets")

# Step 8: Display Frequent Itemsets
# ----------------------------------
print(f"\n🔍 DETAILED FREQUENT ITEMSETS:")

for level, itemsets in enumerate(frequent_itemsets, 1):
    print(f"\n📌 Frequent {level}-itemsets (Top 10):")
    print("-" * 60)
    
    for i, (itemset, support) in enumerate(itemsets[:10], 1):
        items_str = ', '.join(itemset)
        print(f"   {i:2d}. {{{items_str}}} → Support: {support:.4f} ({support*100:.2f}%)")
    
    if len(itemsets) > 10:
        print(f"   ... and {len(itemsets) - 10} more")

# Step 9: Generate Association Rules
# -----------------------------------
def generate_association_rules(frequent_itemsets, transactions, min_confidence=0.6, min_lift=1.0):
    """Generate association rules from frequent itemsets"""
    print(f"\n🎯 Generating Association Rules...")
    print(f"   Min Confidence: {min_confidence*100:.0f}%")
    print(f"   Min Lift: {min_lift}")
    
    rules = []
    
    # Generate rules from itemsets of size 2 and above
    for level in range(1, len(frequent_itemsets)):  # Skip 1-itemsets
        for itemset, itemset_support in frequent_itemsets[level]:
            if len(itemset) < 2:
                continue
            
            # Generate all possible antecedent-consequent combinations
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = tuple([item for item in itemset if item not in antecedent])
                    
                    if not consequent:
                        continue
                    
                    # Calculate confidence
                    antecedent_support = calculate_support(list(antecedent), transactions)
                    if antecedent_support == 0:
                        continue
                    
                    confidence = itemset_support / antecedent_support
                    
                    # Calculate lift
                    consequent_support = calculate_support(list(consequent), transactions)
                    if consequent_support == 0:
                        continue
                    
                    lift = confidence / consequent_support
                    
                    # Filter rules based on confidence and lift
                    if confidence >= min_confidence and lift >= min_lift:
                        rule = {
                            'antecedent': list(antecedent),
                            'consequent': list(consequent),
                            'support': itemset_support,
                            'confidence': confidence,
                            'lift': lift,
                            'conviction': (1 - consequent_support) / (1 - confidence) if confidence < 1 else float('inf')
                        }
                        rules.append(rule)
    
    # Sort rules by confidence (descending)
    rules.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"✅ Generated {len(rules)} association rules")
    return rules

# Generate rules with different confidence thresholds
association_rules = generate_association_rules(frequent_itemsets, transactions, 
                                             min_confidence=0.6, min_lift=1.0)

# Step 10: Display Association Rules
# -----------------------------------
print(f"\n🎯 TOP ASSOCIATION RULES:")
print("="*100)
print(f"{'#':<3} {'Antecedent':<30} {'→':<2} {'Consequent':<25} {'Supp':<6} {'Conf':<6} {'Lift':<6}")
print("="*100)

for i, rule in enumerate(association_rules[:20], 1):
    antecedent_str = ', '.join(rule['antecedent'])
    if len(antecedent_str) > 28:
        antecedent_str = antecedent_str[:25] + '...'
    
    consequent_str = ', '.join(rule['consequent'])
    if len(consequent_str) > 23:
        consequent_str = consequent_str[:20] + '...'
    
    print(f"{i:<3} {antecedent_str:<30} {'→':<2} {consequent_str:<25} "
          f"{rule['support']:.3f} {rule['confidence']:.3f} {rule['lift']:.3f}")

# Step 11: Analyze Specific Rules
# --------------------------------
print(f"\n🔍 DETAILED RULE ANALYSIS:")

# Top rules by confidence
print(f"\n📈 Top 5 Rules by Confidence:")
for i, rule in enumerate(association_rules[:5], 1):
    ant = ' + '.join(rule['antecedent'])
    con = ' + '.join(rule['consequent'])
    print(f"   {i}. IF {ant} THEN {con}")
    print(f"      → Confidence: {rule['confidence']:.3f} ({rule['confidence']*100:.1f}%)")
    print(f"      → Lift: {rule['lift']:.3f}")
    print(f"      → Support: {rule['support']:.3f} ({rule['support']*100:.1f}%)")
    print()

# Rules involving popular items
popular_items = ['whole milk', 'other vegetables', 'rolls/buns', 'soda', 'yogurt']

for item in popular_items[:3]:
    print(f"\n🛒 Rules involving '{item}':")
    item_rules = [rule for rule in association_rules 
                  if item in rule['antecedent'] or item in rule['consequent']]
    
    for i, rule in enumerate(item_rules[:3], 1):
        ant = ' + '.join(rule['antecedent'])
        con = ' + '.join(rule['consequent'])
        print(f"   {i}. {ant} → {con} (Conf: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f})")

# Step 12: Visualize Association Rules
# ------------------------------------
if association_rules:
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Support vs Confidence
    plt.subplot(2, 2, 1)
    supports = [rule['support'] for rule in association_rules[:50]]
    confidences = [rule['confidence'] for rule in association_rules[:50]]
    plt.scatter(supports, confidences, alpha=0.6, color='blue')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (Top 50 Rules)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Confidence vs Lift
    plt.subplot(2, 2, 2)
    lifts = [rule['lift'] for rule in association_rules[:50]]
    plt.scatter(confidences, lifts, alpha=0.6, color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Lift')
    plt.title('Confidence vs Lift (Top 50 Rules)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Rule Distribution by Confidence
    plt.subplot(2, 2, 3)
    plt.hist([rule['confidence'] for rule in association_rules], bins=20, 
             color='green', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Number of Rules')
    plt.title('Distribution of Rule Confidence')
    
    # Plot 4: Rule Distribution by Lift
    plt.subplot(2, 2, 4)
    plt.hist([rule['lift'] for rule in association_rules if rule['lift'] <= 10], 
             bins=20, color='orange', alpha=0.7)
    plt.xlabel('Lift')
    plt.ylabel('Number of Rules')
    plt.title('Distribution of Rule Lift')
    
    plt.tight_layout()
    plt.show()

# Step 13: Business Insights and Recommendations
# -----------------------------------------------
print(f"\n💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*80)

# Market basket insights
print(f"\n🛒 MARKET BASKET INSIGHTS:")
print(f"   • Average basket size: {np.mean(transaction_lengths):.2f} items")
print(f"   • Most transactions (mode): {max(set(transaction_lengths), key=transaction_lengths.count)} items")
print(f"   • {len([t for t in transaction_lengths if t == 1])} single-item transactions ({len([t for t in transaction_lengths if t == 1])/len(transactions)*100:.1f}%)")

# Strong associations
strong_rules = [rule for rule in association_rules if rule['lift'] > 2.0 and rule['confidence'] > 0.8]
print(f"\n🎯 STRONG ASSOCIATIONS (Lift > 2.0, Confidence > 80%):")
for rule in strong_rules[:5]:
    ant = ' + '.join(rule['antecedent'])
    con = ' + '.join(rule['consequent'])
    print(f"   • {ant} → {con}")
    print(f"     Customers who buy {ant} have {rule['confidence']*100:.1f}% chance of also buying {con}")

# Recommendations
print(f"\n💼 BUSINESS RECOMMENDATIONS:")

print(f"\n1. 🏪 STORE LAYOUT OPTIMIZATION:")
print(f"   • Place 'whole milk' in a strategic location (appears in {item_counts['whole milk']/len(transactions)*100:.1f}% of transactions)")
print(f"   • Position frequently associated items near each other")
print(f"   • Create end-cap displays with complementary products")

print(f"\n2. 🎯 CROSS-SELLING STRATEGIES:")
if association_rules:
    top_rule = association_rules[0]
    ant = ' + '.join(top_rule['antecedent'])
    con = ' + '.join(top_rule['consequent'])
    print(f"   • When customers buy {ant}, suggest {con}")
    print(f"   • Success rate: {top_rule['confidence']*100:.1f}%")

print(f"\n3. 📦 INVENTORY MANAGEMENT:")
top_5 = [item for item, count in top_20_items[:5]]
print(f"   • Prioritize stock for: {', '.join(top_5)}")
print(f"   • Use association rules to predict demand spikes")

print(f"\n4. 🎉 PROMOTIONAL OPPORTUNITIES:")
print(f"   • Bundle frequently bought items together")
print(f"   • Offer discounts on antecedent items to boost consequent sales")
print(f"   • Target customers with personalized recommendations")

print(f"\n5. 📊 PERFORMANCE METRICS:")
print(f"   • Monitor basket size increases after implementing recommendations")
print(f"   • Track conversion rates for suggested item combinations")
print(f"   • Measure revenue per transaction improvements")

# Step 14: Export Results (Optional)
# -----------------------------------
print(f"\n💾 EXPORT RESULTS:")

# Create DataFrame for rules
rules_df = pd.DataFrame([
    {
        'Rule': ' + '.join(rule['antecedent']) + ' → ' + ' + '.join(rule['consequent']),
        'Antecedent': ', '.join(rule['antecedent']),
        'Consequent': ', '.join(rule['consequent']),
        'Support': rule['support'],
        'Confidence': rule['confidence'],
        'Lift': rule['lift']
    }
    for rule in association_rules
])

# Create DataFrame for frequent items
items_df = pd.DataFrame([
    {
        'Item': item,
        'Frequency': count,
        'Support': count / len(transactions)
    }
    for item, count in top_20_items
])

print(f"   ✅ Created rules DataFrame with {len(rules_df)} rules")
print(f"   ✅ Created items DataFrame with {len(items_df)} items")

# Display sample of results
print(f"\n📋 SAMPLE RESULTS:")
print(f"\nTop 5 Association Rules:")
print(rules_df.head())

print(f"\nTop 10 Frequent Items:")
print(items_df.head(10))

print(f"\n🎉 MARKET BASKET ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"   • Analyzed {len(transactions):,} transactions")
print(f"   • Found {len(association_rules)} association rules")
print(f"   • Identified {sum(len(itemsets) for itemsets in frequent_itemsets)} frequent itemsets")

# Optional: Save to CSV
# rules_df.to_csv('association_rules.csv', index=False)
# items_df.to_csv('frequent_items.csv', index=False)
# print(f"   • Results saved to CSV files")