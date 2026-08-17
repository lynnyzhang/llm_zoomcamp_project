import streamlit as st

from src.rag.tools import SearchRecord
from src.search.search_records import PokemonDoc, TypeChartDoc


class CardRenderer:
    """Pure rendering of Pokémon cards; no session state here."""

    def pokemon_doc(
        self, searches: list[SearchRecord], question: str
    ) -> list[PokemonDoc]:
        """Docs for Pokémon named in the question, deduped by id — the identity
        key shared by Pokémon and docs (alternate forms have distinct ids)."""
        question_lower = question.lower()
        seen_ids = set()
        docs = []

        for search in searches:
            for doc in search.results:
                if not isinstance(doc, PokemonDoc):
                    continue
                doc_id = doc.id
                name = doc.name
                if name and name.lower() in question_lower and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append(doc)

        return docs

    def card_title(self, doc: PokemonDoc | TypeChartDoc) -> str:
        if isinstance(doc, TypeChartDoc):
            return f"{doc.type or 'Unknown'} type chart"
        name = doc.name
        doc_id = doc.id
        if isinstance(doc_id, int):
            return f"{name} (#{doc_id})"
        return name

    def card_caption(self, doc: PokemonDoc | TypeChartDoc) -> str:
        if isinstance(doc, TypeChartDoc):
            return "Type chart"
        types = doc.types
        if types:
            return " + ".join(types)
        return "unknown"

    def pokemon_card_grid(self, docs: list[PokemonDoc | TypeChartDoc]):
        st.subheader("Pokémon Cards")

        for row_start in range(0, len(docs), 4):
            row_docs = docs[row_start : row_start + 4]
            columns = st.columns(4)
            for col, doc in zip(columns, row_docs):
                with col:
                    st.markdown(f"**{self.card_title(doc)}**")
                    caption = self.card_caption(doc)
                    if caption:
                        st.caption(caption)
