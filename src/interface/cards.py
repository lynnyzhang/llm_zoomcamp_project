import contextlib
import re

import streamlit as st


class CardRenderer:
    # Pure rendering of Pokémon cards; no session state here.

    def pokemon_doc(self, searches, question):
        """Docs for Pokémon named in the question, deduped by id — the identity
        key shared by Pokémon and docs (alternate forms have distinct ids)."""
        question_lower = question.lower()
        seen_ids = set()
        docs = []

        for search in searches:
            for doc in search.results:
                # Only doc-shaped local results render as cards; web results
                # (title/url/snippet) carry no "id" and must never be shown.
                if not isinstance(doc, dict) or "id" not in doc:
                    continue
                doc_id = doc.get("id", "")
                name = doc.get("name", "")
                if name and name.lower() in question_lower and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append(doc)

        return docs

    def doc_artwork_url(self, doc):
        doc_id = str(doc.get("id", ""))
        pokemon_id = re.sub(r"\D", "", doc_id)
        if pokemon_id:
            # High-res 475x475 official artwork — the dataset sprite_url is a
            # small 96x96 image. Missing artwork (404) degrades to no image
            # via the caller's suppress.
            return (
                "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                f"sprites/pokemon/other/official-artwork/{pokemon_id}.png"
            )
        # Non-numeric ids (type-chart docs) have no artwork; keep any sprite.
        return doc.get("sprite_url") or ""

    def stats_summary(self, doc, limit=200):
        # Type-chart docs carry no stats.
        if doc.get("kind") == "type_chart":
            return ""
        stats = doc.get("stats") or {}
        if not stats:
            return ""
        parts = [
            f"hp {stats.get('hp', 0)}",
            f"attack {stats.get('attack', 0)}",
            f"defense {stats.get('defense', 0)}",
            f"sp. attack {stats.get('sp_attack', 0)}",
            f"sp. defense {stats.get('sp_defense', 0)}",
            f"speed {stats.get('speed', 0)}",
        ]
        return ", ".join(parts)[:limit]

    def card_title(self, doc):
        if doc.get("kind") == "type_chart":
            return f"{doc.get('type') or 'Unknown'} type chart"
        name = doc.get("name", "Untitled")
        doc_id = doc.get("id")
        if isinstance(doc_id, int):
            return f"{name} (#{doc_id})"
        return name

    def card_caption(self, doc):
        if doc.get("kind") == "type_chart":
            return "Type chart"
        types = doc.get("types") or []
        if types:
            return " + ".join(types)
        return "unknown"

    def pokemon_card_grid(self, docs):
        st.subheader("Pokémon Cards")

        for row_start in range(0, len(docs), 4):
            row_docs = docs[row_start:row_start + 4]
            columns = st.columns(4)
            for col, doc in zip(columns, row_docs):
                with col:
                    title = self.card_title(doc)
                    artwork_url = self.doc_artwork_url(doc)
                    if artwork_url:
                        # Broken/404 artwork must not break the card — the
                        # title and stats below still render without it.
                        with contextlib.suppress(Exception):
                            st.image(artwork_url, width="stretch")
                    st.markdown(f"**{title}**")
                    caption = self.card_caption(doc)
                    if caption:
                        st.caption(caption)
                    summary = self.stats_summary(doc)
                    if summary:
                        st.caption(summary)
