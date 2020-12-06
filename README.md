# ingredient-generator
Character-prediction RNN for creating novel recipe ingredients

Generates "recipe material" or ingredients that would be typical of a recipe.

Training examples are of the form:

* Amount (fraction string such as "1 1/2" - can be null)
* Measure (abbreviated measurement type such as "tbsp" - can be null)
* Type / size (e.g., "small" or "roasted" - can be null)
* Ingredient (e.g., "black pepper" - not null)
* Preparation (e.g., "stemmed and seeded" - can be null and is preceded by comma if exists)

Training examples, and intended results, look like "1 1/2 tbsp lemon juice, freshly squeezed"
