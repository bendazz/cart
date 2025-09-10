# CART Visualization Project

This project will provide a web application to illustrate the step-by-step construction of a Decision Tree using the CART algorithm (Classification and Regression Trees) on a small, interpretable subset of the classic Iris dataset.

## Current Status

- Environment dependencies declared in `requirements.txt`.
- Deterministic 20-sample stratified subset of the Iris dataset generated at `data/iris_subset.json`.

## Iris Subset Details

We select 20 samples (roughly balanced across the three Iris classes) using a deterministic random seed so the subset is reproducible.

Script: `data/select_iris_subset.py`

Output file structure (`data/iris_subset.json`):

```
{
	feature_names: [string, ...],
	target_names: [string, ...],
	samples: [
		{
			index: <int original index>,
			features: [sepal_length, sepal_width, petal_length, petal_width],
			target: <class_id>,
			target_name: <class_label>
		},
		...
	],
	total_samples: 20,
	random_seed: 42
}
```

## Re-generate the Subset

If you need to recreate the subset (will overwrite the JSON file):

```
python data/select_iris_subset.py
```

## Next Steps (Planned)

1. Implement a backend endpoint to serve the subset and progressive CART splitting steps.
2. Implement a pure-Python CART step engine that records impurity calculations (Gini) and candidate splits.
3. Build an interactive frontend (likely React or lightweight JS) to visualize:
	 - Current node sample distribution
	 - Candidate splits and chosen split rationale
	 - Tree structure growth over iterations
4. Add tests for deterministic split selection and impurity calculations.

## Frontend Feature Projection (Current Focus)

The interface has been simplified to focus solely on a feature projection visualization of the 20-sample Iris subset.

Included assets:

- `index.html` main page with a single panel.
- `styles.css` layout, circles, scales.
- `js/app.js` loads the subset and renders a horizontal number line for a selected feature with colored circles (one per sample) using a deterministic color per class.
- `data/iris_subset.json` dataset subset.

You can switch the feature via a dropdown; the samples are sorted and positioned proportionally between min and max (with 5% padding). A legend shows class-color mapping.

Serve the static files (examples):
```
python -m http.server 8000
```
Browse: http://localhost:8000/

Or:
```
npx serve .
```

The previous experimental CART step-by-step UI has been removed for now; `js/cart_core.js` can be reintroduced later if tree visualization is needed again.

## Development Setup

Install dependencies:

```
pip install -r requirements.txt
```

## License

MIT (add a LICENSE file if distributing).
