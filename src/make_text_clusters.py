from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def load_config(filename: str):
	return OmegaConf.load(filename)

@main(version_base=None, config_path="../configs", config_name="make_text_clusters.yaml")
def _main(cfg: DictConfig):
	"""
	Text pseudo-classes for conditional DD: TF-IDF of character n-grams (2-3) over the
	transcriptions of the pool + k-means. The hypothesis: unconditional matching averages
	gradients of heterogeneous sequences whose content components cancel out; conditioning
	on a text cluster (analogous to "class = same content" in classification DD, reinforced
	in IAM because several writers copy the same LOB text) turns the matching objective into
	something with content.

	Output (YAML): `clusters` (cluster id per POSITION in the filtered pool -- the same
	positional order used by distill_{dc,dm}_ctc.py) and `medoids` (position of the line
	closest to the centroid of each cluster, in TF-IDF space).
	"""
	pool = instantiate(cfg.real_dataset)
	texts = ["".join(pool.int_to_class[t] for t in pool[p]["label"].tolist()) for p in range(len(pool))]
	print(f"Pool: {len(texts)} líneas; k={cfg.k}")

	vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
	X = vectorizer.fit_transform(texts)

	kmeans = KMeans(n_clusters=int(cfg.k), random_state=int(cfg.seed), n_init=10)
	assignments = kmeans.fit_predict(X)

	medoids: list[int] = []
	for c in range(int(cfg.k)):
		members = [p for p, a in enumerate(assignments) if a == c]
		assert members, f"Cluster {c} vacío — reduce k o cambia la semilla."
		diff = X[members].toarray() - kmeans.cluster_centers_[c]
		distances = (diff ** 2).sum(axis=1)
		medoids.append(members[int(distances.argmin())])

	sizes = [int((assignments == c).sum()) for c in range(int(cfg.k))]
	print(f"Tamaños de cluster: min={min(sizes)} mediana={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")
	for c in range(min(3, int(cfg.k))):
		print(f"  cluster {c} (n={sizes[c]}), medoide: '{texts[medoids[c]][:60]}'")

	OmegaConf.save(
			OmegaConf.create({
				"k": int(cfg.k),
				"clusters": [int(a) for a in assignments],
				"medoids": medoids,
				}),
			cfg.output_file,
			)
	print(f"Guardado en '{cfg.output_file}'.")

if __name__ == "__main__":
	OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", load_config)
	_main()
