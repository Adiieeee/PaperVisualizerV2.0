"""
Corpus manager — kelola korpus paper tanpa perlu build ulang.

Perintah:
    python corpus.py list                  # daftar semua backup korpus
    python corpus.py backup <nama-tema>    # simpan korpus aktif ke backups/<nama-tema>/
    python corpus.py clear                 # kosongkan korpus aktif (siap diisi paper baru)
    python corpus.py restore <nama-tema>   # pulihkan korpus dari backup (tanpa build ulang)
    python corpus.py manifest              # tulis manifest integritas untuk korpus aktif (tanpa build ulang)
    python corpus.py audit                 # pindai graph + chunk untuk teks mencurigakan (read-only)

Contoh alur ganti tema paper:
    python corpus.py backup cnn-klasik     # simpan yang sekarang
    python corpus.py clear                 # bersihkan
    # ... taruh PDF baru di data/raw/ ...
    python main.py build
"""
import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
BACKUPS = Path("backups")
EVAL_CASES = Path("eval/eval_cases.json")

# File hasil build yang menentukan apakah korpus bisa dipakai tanpa build ulang.
# manifest.json ikut di-backup: tanpa dia, engine tidak bisa memverifikasi korpus hasil restore.
ESSENTIAL = ["graph.graphml", "paper_metadata.json", "manifest.json"]   # + folder chroma_db

# Nama backup dipakai sebagai nama folder (dan bisa memicu rmtree saat menimpa): tolak pemisah path / "..".
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name)) and ".." not in name
OPTIONAL = ["graph_visualization.html"]
# Sengaja TIDAK di-backup: similarity_debug*.csv (log diagnostik, besar, tidak dipakai runtime)


def _dir_size(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _graph_stats(graphml: Path):
    """Baca jumlah node/edge tanpa dependensi berat (parsing XML sederhana)."""
    if not graphml.exists():
        return None
    try:
        text = graphml.read_text(encoding="utf-8", errors="ignore")
        return text.count("<node "), text.count("<edge ")
    except Exception:
        return None


def cmd_list(_args):
    if not BACKUPS.exists() or not any(BACKUPS.iterdir()):
        print("Belum ada backup. Buat dengan: python corpus.py backup <nama-tema>")
        return
    print(f"{'NAMA':28} {'PDF':>4} {'NODE':>6} {'EDGE':>6}  UKURAN")
    print("-" * 62)
    for d in sorted(p for p in BACKUPS.iterdir() if p.is_dir()):
        pdfs = len(list((d / "raw").glob("*.pdf"))) if (d / "raw").exists() else 0
        stats = _graph_stats(d / "processed" / "graph.graphml")
        n, e = stats if stats else ("-", "-")
        print(f"{d.name:28} {pdfs:>4} {n:>6} {e:>6}  {_dir_size(d)}")


def cmd_backup(args):
    name = args.name.strip().replace(" ", "-")
    if not _valid_name(name):
        print(f"[!] Nama backup '{name}' tidak valid. Pakai huruf/angka/titik/strip/garis-bawah saja "
              f"(tanpa '/', '\\' atau '..'), diawali huruf/angka.")
        return
    dest = BACKUPS / name
    if dest.exists():
        if input(f"Backup '{name}' sudah ada. Timpa? [y/N] ").strip().lower() != "y":
            print("Dibatalkan.")
            return
        shutil.rmtree(dest)

    if not (PROCESSED / "graph.graphml").exists():
        print("[!] data/processed/graph.graphml tidak ada — tidak ada korpus untuk di-backup.")
        print("    Jalankan 'python main.py build' dulu.")
        return

    (dest / "raw").mkdir(parents=True, exist_ok=True)
    (dest / "processed").mkdir(parents=True, exist_ok=True)

    pdfs = list(RAW.glob("*.pdf"))
    for f in pdfs:
        shutil.copy2(f, dest / "raw" / f.name)

    for fname in ESSENTIAL + OPTIONAL:
        src = PROCESSED / fname
        if src.exists():
            shutil.copy2(src, dest / "processed" / fname)

    if (PROCESSED / "chroma_db").exists():
        shutil.copytree(PROCESSED / "chroma_db", dest / "processed" / "chroma_db")

    if EVAL_CASES.exists():
        shutil.copy2(EVAL_CASES, dest / "eval_cases.json")

    # Catatan ringkas supaya isi backup jelas saat dibuka lagi berbulan-bulan kemudian.
    stats = _graph_stats(dest / "processed" / "graph.graphml")
    meta_path = dest / "processed" / "paper_metadata.json"
    titles = []
    if meta_path.exists():
        try:
            titles = [v.get("title", "?") for v in json.loads(meta_path.read_text(encoding="utf-8")).values()]
        except Exception:
            pass
    lines = [
        f"# Backup Korpus: {name}",
        "",
        f"- Tanggal: {date.today().isoformat()}",
        f"- Jumlah PDF: {len(pdfs)}",
    ]
    if stats:
        lines.append(f"- Graph: {stats[0]} node, {stats[1]} edge")
    if titles:
        lines += ["", "## Paper", *[f"- {t}" for t in titles]]
    lines += [
        "",
        "## Restore",
        "",
        "```powershell",
        f"python corpus.py restore {name}",
        "```",
        "",
        "Setelah restore, langsung bisa dipakai tanpa build ulang:",
        "`python server.py` atau `python main.py query --query \"...\"`.",
    ]
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Korpus di-backup ke {dest}  ({_dir_size(dest)}, {len(pdfs)} PDF)")


def _wipe_current(verbose=True):
    removed = []
    for f in RAW.glob("*.pdf"):
        f.unlink(); removed.append(str(f))
    for pattern in ("graph.graphml", "paper_metadata.json", "graph_visualization.html",
                    "similarity_debug*.csv", "checkpoint_*.json"):
        for f in PROCESSED.glob(pattern):
            f.unlink(); removed.append(str(f))
    chroma = PROCESSED / "chroma_db"
    if chroma.exists():
        shutil.rmtree(chroma); removed.append(str(chroma))
    if verbose:
        for r in removed:
            print(f"  - {r}")
        print(f"[OK] {len(removed)} item dihapus. Siap diisi PDF baru di data/raw/")
    return removed


def cmd_clear(args):
    has_graph = (PROCESSED / "graph.graphml").exists()
    pdfs = list(RAW.glob("*.pdf"))
    if not has_graph and not pdfs:
        print("Korpus aktif sudah kosong.")
        return

    print("Akan MENGHAPUS korpus aktif:")
    print(f"  - {len(pdfs)} PDF di data/raw/")
    print("  - graph.graphml, paper_metadata.json, chroma_db/, visualisasi, log & checkpoint")
    if not args.yes:
        if not (PROCESSED / "graph.graphml").exists():
            print("\n[!] Tidak ada graph.graphml — tidak ada hasil build yang hilang.")
        else:
            print("\n[!] Belum di-backup? Batalkan lalu jalankan: python corpus.py backup <nama-tema>")
        if input("Lanjut hapus? [y/N] ").strip().lower() != "y":
            print("Dibatalkan.")
            return
    _wipe_current()


def cmd_restore(args):
    if not _valid_name(args.name):
        print(f"[!] Nama backup '{args.name}' tidak valid (tanpa '/', '\\' atau '..').")
        return
    src = BACKUPS / args.name
    if not src.exists():
        print(f"[!] Backup '{args.name}' tidak ditemukan di {BACKUPS}/")
        cmd_list(args)
        return

    if (PROCESSED / "graph.graphml").exists() and not args.yes:
        print("[!] Ada korpus aktif yang akan ditimpa.")
        print("    Simpan dulu dengan: python corpus.py backup <nama-tema>")
        if input("Lanjut restore (korpus aktif dihapus)? [y/N] ").strip().lower() != "y":
            print("Dibatalkan.")
            return

    _wipe_current(verbose=False)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    for f in (src / "raw").glob("*.pdf"):
        shutil.copy2(f, RAW / f.name)
    for f in (src / "processed").glob("*"):
        if f.is_dir():
            shutil.copytree(f, PROCESSED / f.name)
        else:
            shutil.copy2(f, PROCESSED / f.name)
    if (src / "eval_cases.json").exists():
        EVAL_CASES.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "eval_cases.json", EVAL_CASES)

    stats = _graph_stats(PROCESSED / "graph.graphml")
    print(f"[OK] Korpus '{args.name}' dipulihkan" + (f" ({stats[0]} node, {stats[1]} edge)" if stats else ""))
    _verify_after_restore()
    print("     Siap dipakai TANPA build ulang: python server.py")
    print("     (restart server.py kalau sedang berjalan — graph di-cache saat startup)")


def _verify_after_restore():
    """Bandingkan hasil restore dengan manifest-nya. Backup lama tanpa manifest hanya diberi peringatan."""
    from src.security.integrity import ArtifactIntegrityError, check_artifacts
    graphml = PROCESSED / "graph.graphml"
    if not graphml.exists():
        return
    try:
        import networkx as nx
        from src.vectorstore.chromadb_store import ChromaStore
        check_artifacts(str(graphml), nx.read_graphml(graphml), ChromaStore())
    except ArtifactIntegrityError as e:
        print(f"[!] Korpus dipulihkan tetapi TIDAK konsisten dengan manifest-nya:\n{e}")


def cmd_manifest(_args):
    """Tulis manifest untuk korpus aktif (migrasi korpus yang dibangun sebelum fitur ini; tanpa build ulang)."""
    import networkx as nx
    from src import config
    from src.security.integrity import MANIFEST_PATH, build_manifest, write_manifest
    from src.vectorstore.chromadb_store import ChromaStore

    graphml = PROCESSED / "graph.graphml"
    if not graphml.exists() or not (PROCESSED / "chroma_db").exists():
        print("[!] graph.graphml / chroma_db tidak ada — tidak ada korpus aktif. Jalankan 'python main.py build'.")
        return
    manifest = build_manifest(str(graphml), nx.read_graphml(graphml), ChromaStore())
    write_manifest(manifest, MANIFEST_PATH)
    g, v, e = manifest["graph"], manifest["vectorstore"], manifest["embedding"]
    print(f"[OK] {MANIFEST_PATH} ditulis: {g['nodes']} node, {g['edges']} edge, {v['chunks']} chunk, "
          f"{v['community_summaries']} ringkasan komunitas, dimensi vektor {e['dimension']}.")
    print(f"     Dicatat sebagai dibangun oleh model '{config.EMBEDDING_MODEL_NAME}' revisi "
          f"{config.EMBEDDING_MODEL_REVISION[:10]}… — ini ASUMSI dari config sekarang.")
    print("     Jalankan perintah ini hanya bila Anda yakin korpus aktif memang dibangun dengan model tersebut.")


def cmd_audit(_args):
    """Read-only: pindai graph + chunk untuk teks yang mencurigakan. Tidak mengubah apa pun."""
    from typing import get_args
    import networkx as nx
    from src import config
    from src.security.sanitize import MAX_ENTITY_NAME_LENGTH, audit_text

    findings = []  # (lokasi, cuplikan, alasan)

    def add(where, text, reasons):
        if reasons:
            findings.append((where, (text or "")[:70].replace("\n", " "), "; ".join(reasons)))

    graphml = PROCESSED / "graph.graphml"
    n_nodes = n_edges = n_chunks = 0
    if graphml.exists():
        G = nx.read_graphml(graphml)
        known_types = set(get_args(config.ENTITY_TYPES))
        known_rels = set(get_args(config.RELATION_TYPES)) | {"variant_of"}
        for n, d in G.nodes(data=True):
            n_nodes += 1
            add(f"node '{str(n)[:30]}'", n, audit_text(n, MAX_ENTITY_NAME_LENGTH))
            add(f"deskripsi node '{str(n)[:30]}'", d.get("description", ""), audit_text(d.get("description", ""), 1000))
            if d.get("type") not in known_types:
                add(f"node '{str(n)[:30]}'", str(d.get("type")), [f"tipe node tak dikenal: {d.get('type')!r}"])
        for u, v, d in G.edges(data=True):
            n_edges += 1
            add(f"deskripsi edge '{str(u)[:20]}'-'{str(v)[:20]}'", d.get("description", ""),
                audit_text(d.get("description", ""), 1000))
            if d.get("relation") not in known_rels:
                add(f"edge '{str(u)[:20]}'-'{str(v)[:20]}'", str(d.get("relation")),
                    [f"relasi tak dikenal: {d.get('relation')!r}"])
    else:
        print("[!] graph.graphml tidak ada — hanya chunk yang diperiksa.")

    if (PROCESSED / "chroma_db").exists():
        from src.vectorstore.chromadb_store import ChromaStore
        store = ChromaStore()
        docs = store.chunks_collection.get(include=["documents"])
        for cid, text in zip(docs.get("ids", []), docs.get("documents", [])):
            n_chunks += 1
            add(f"chunk {cid}", text, audit_text(text, names=False))
        summaries = store.community_collection.get(include=["documents"])
        for cid, text in zip(summaries.get("ids", []), summaries.get("documents", [])):
            add(f"ringkasan komunitas {cid}", text, audit_text(text, names=False))

    print(f"Diperiksa: {n_nodes} node, {n_edges} edge, {n_chunks} chunk.")
    if not findings:
        print("[OK] Tidak ada temuan mencurigakan.")
        return
    print(f"[!] {len(findings)} temuan (hanya laporan; tidak ada yang diubah):")
    for where, snippet, why in findings[:100]:
        print(f"  - {where}: {why}\n      \"{snippet}\"")
    if len(findings) > 100:
        print(f"  ... dan {len(findings) - 100} lainnya.")


def main():
    ap = argparse.ArgumentParser(description="Kelola korpus paper (backup / clear / restore).")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="daftar backup korpus")

    p_b = sub.add_parser("backup", help="simpan korpus aktif ke backups/<nama>")
    p_b.add_argument("name", help="nama tema, mis. cnn-klasik / nlp-transformer")

    p_c = sub.add_parser("clear", help="kosongkan korpus aktif")
    p_c.add_argument("-y", "--yes", action="store_true", help="tanpa konfirmasi")

    p_r = sub.add_parser("restore", help="pulihkan korpus dari backup")
    p_r.add_argument("name", help="nama backup")
    p_r.add_argument("-y", "--yes", action="store_true", help="tanpa konfirmasi")

    sub.add_parser("manifest", help="tulis manifest integritas untuk korpus aktif (tanpa build ulang)")
    sub.add_parser("audit", help="pindai graph + chunk untuk teks mencurigakan (read-only)")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        sys.exit(1)
    {"list": cmd_list, "backup": cmd_backup, "clear": cmd_clear, "restore": cmd_restore,
     "manifest": cmd_manifest, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
