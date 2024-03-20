import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class FeatureGrouper:
    def __init__(self, options_path, threshold=0.8):
        self.options_path = options_path
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        self.options = self._load_and_preprocess_options()
        self.tfidf_matrix = self._calculate_tfidf()
        self.cosine_similarities = self._calculate_cosine_similarity()
        self.groups = None
        self.output_df = None

    def _load_and_preprocess_options(self):
        with open(self.options_path, "r", encoding="utf-8") as file:
            options = file.readlines()
        options = [option.strip() for option in options if option.strip()]
        return options

    @staticmethod
    def _remove_literal_duplicates(option):
        return re.sub(r"[_ ][\dabcdABCD]+$", "", option)

    def _calculate_tfidf(self):
        return self.vectorizer.fit_transform(
            list(map(self._remove_literal_duplicates, self.options))
        )

    def _calculate_cosine_similarity(self):
        return cosine_similarity(self.tfidf_matrix)

    def group_options(self):
        self.groups = self._group_options()
        self.output_df = self._prepare_output_dataframe()

    def _group_options(self):
        groups = {}
        for i, row in enumerate(self.cosine_similarities):
            for j, similarity in enumerate(row):
                if i != j and similarity > self.threshold:
                    if i not in groups and j not in groups:
                        groups[i] = len(groups)
                        groups[j] = groups[i]
                    elif i in groups and j not in groups:
                        groups[j] = groups[i]
                    elif j in groups and i not in groups:
                        groups[i] = groups[j]
        return groups

    def _prepare_output_dataframe(self):
        output_df = pd.DataFrame(
            {
                "Option": self.options,
                "Group": [
                    self.groups.get(i, -1) for i in range(len(self.options))
                ],
            }
        )
        group_names = self._assign_group_names(output_df)
        output_df["GroupName"] = output_df["Group"].map(group_names)
        output_df["GroupName"] = output_df["GroupName"].apply(
            self._simplify_option
        )
        # Rename the ungrouped phenomena to their orignal names
        output_df.loc[output_df["Group"] == -1, "GroupName"] = output_df.loc[
            output_df["Group"] == -1, "Option"
        ]
        output_df = output_df.sort_values("Group", ascending=False)
        return output_df

    @staticmethod
    def _assign_group_names(df):
        group_names = {}
        for index, row in df.iterrows():
            group = row["Group"]
            option = row["Option"]
            if group not in group_names:
                group_names[group] = option
        return group_names

    @staticmethod
    def _simplify_option(option):
        return re.sub(r"^[^a-zA-Z]+ | [^a-zA-Z\s\W]+$", "", option).strip()

    def save_to_csv(self, path):
        if self.output_df is not None:
            self.output_df.to_csv(path, index=False)
        else:
            raise ValueError("Group options before saving to CSV.")

    def get_summary(self):
        if self.output_df is not None:
            n_unclassified = len(
                self.output_df.loc[self.output_df.Group == -1]
            )
            after_amount = n_unclassified + len(
                self.output_df.loc[
                    self.output_df.Group != -1, "Group"
                ].unique()
            )
            return f"Original amount of phenomena: {len(self.options)}\nWith TF-IDF deduplicating the amount of phenomena is reduced to: {after_amount}"


if __name__ == "__main__":
    grouper = FeatureGrouper("all_features.txt", threshold=0.7)
    grouper.group_options()
    output_path = "grouped_options.csv"
    grouper.save_to_csv(output_path)
    print(grouper.get_summary())
    print("Find the results in", output_path)
