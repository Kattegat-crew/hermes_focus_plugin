"""El recibo debe reportar el backend CONFIGURADO también cuando el turno no inyecta nada.

En la ruta ``DIRECT_REPLY`` el coordinador retorna antes de asignar
``configured_router_backend``, así que el recibo mostraba el default del dataclass
(``host_llm``) aunque el usuario hubiera configurado ``graph``. Es una discrepancia de
reporte: el recibo debe describir la configuración real, no un valor por defecto.
"""
import pytest

from neural_context.config import PluginConfig
from neural_context.models import CognitiveRoute, TriageAnalysis


class _StubSidecar:
    n_docs = 0
    total_catalog_tokens = 0
    db_path = ":memory:"

    def sync_skills(self, *a, **k):
        return None

    def filter_query_terms(self, text, min_weight=0.25):
        return []

    def search_bm25(self, query, top_k=0):
        return []


@pytest.mark.parametrize("backend", ["graph", "host_llm"])
def test_direct_reply_reports_configured_backend(backend, tmp_path):
    from neural_context.router import ContextCoordinator

    config = PluginConfig(router_backend=backend, software_inspection_enabled=False)
    coordinator = ContextCoordinator(config=config, sidecar=_StubSidecar(), ctx=None)

    analysis = TriageAnalysis(
        route=CognitiveRoute.DIRECT_REPLY,
        is_actionable_task=False,
        needs_skills=False,
        needs_memory=False,
        detected_language="es",
    )

    result = coordinator.execute(analysis, user_message="hola, ¿cómo estás?", cwd=str(tmp_path))

    assert result.status == "skipped_no_context_needed"
    assert result.router_backend is None, "no corrió ningún backend: no debe reportarse uno"
    assert result.configured_router_backend == backend
