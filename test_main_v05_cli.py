import json
import os
import subprocess
import sys
import tempfile


REF_DIR = r"data\reference_TheWorld50"
TEST_IMAGE = r"data\reference_TheWorld50\bridge\bridge_01.png"


def fail(message: str):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str):
    print(f"[OK] {message}")


def run_cmd(args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        fail(f"Команда завершилась с ошибкой: {' '.join(args)}")

    return result.stdout


def main():
    print("=== TEST MAIN V0.5 CLI ===")

    if not os.path.isdir(REF_DIR):
        fail(f"Папка эталонов не найдена: {REF_DIR}")

    if not os.path.isfile(TEST_IMAGE):
        fail(f"Тестовая картинка не найдена: {TEST_IMAGE}")

    ok("Исходные данные найдены")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db_v05.json")

        build_out = run_cmd([
            sys.executable,
            "main_v05.py",
            "build_db_v05",
            "--ref-dir", REF_DIR,
            "--db", db_path,
            "--preprocess-mode", "none",
            "--threshold-mode", "quantile",
            "--descriptor-mode", "texture100",
        ])

        if "vector_length: 152" not in build_out:
            print("STDOUT build_db_v05:")
            print(build_out)
            fail("В выводе build_db_v05 нет vector_length 152")

        ok("build_db_v05 отработал без ошибки")

        if not os.path.isfile(db_path):
            fail("Файл тестовой базы не создан")

        with open(db_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("meta", {})
        records = payload.get("records", [])

        if meta.get("version") != "v0.5":
            fail(f"Некорректная версия в meta: {meta.get('version')}")

        if meta.get("descriptor_mode") != "texture100":
            fail(f"Некорректный descriptor_mode в meta: {meta.get('descriptor_mode')}")

        if meta.get("threshold_mode") != "quantile":
            fail(f"Некорректный threshold_mode в meta: {meta.get('threshold_mode')}")

        if int(meta.get("vector_length", 0)) != 152:
            fail(f"Некорректный vector_length в meta: {meta.get('vector_length')}")

        if len(records) != 80:
            fail(f"Ожидалось 80 records, получено: {len(records)}")

        first_vector_len = len(records[0].get("vector", []))
        if first_vector_len != 152:
            fail(f"Ожидалась длина первого vector 152, получено: {first_vector_len}")

        ok("JSON-база содержит корректную meta и records")

        match_out = run_cmd([
            sys.executable,
            "main_v05.py",
            "match_v05",
            "--image", TEST_IMAGE,
            "--db", db_path,
            "--preprocess-mode", "none",
            "--threshold-mode", "quantile",
            "--descriptor-mode", "texture100",
            "--top-k", "5",
        ])

        if "=== V0.5 QUERY DESCRIPTOR ===" not in match_out:
            fail("В выводе match_v05 нет блока V0.5 QUERY DESCRIPTOR")

        if "bars:" not in match_out:
            fail("В выводе match_v05 нет количества bars")

        if "vector_length: 152" not in match_out:
            fail("В выводе match_v05 нет vector_length 152")

        if "1. bridge_01 | class=bridge | distance=0.0 | similarity=1.0" not in match_out:
            print(match_out)
            fail("Top-1 не вернул bridge_01 с distance=0.0")

        ok("match_v05 корректно вернул bridge_01 в Top-1")

    print("")
    print("=== RESULT ===")
    print("main_v05.py прошел CLI-тест. Рабочая версия v0.5 texture100 готова.")


if __name__ == "__main__":
    main()