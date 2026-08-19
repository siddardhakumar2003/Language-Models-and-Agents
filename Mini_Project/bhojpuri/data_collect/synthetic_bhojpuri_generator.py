#!/usr/bin/env python3
"""
Synthetic Bhojpuri Text Generator Orchestrator.
Resumable, background-runnable cycle-based generator (like Phase 3 translation).
"""
import json
import logging
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Topic diversification per worker
WORKER_TOPICS = {
    'a': {
        'name': 'Folk Tales & Cultural Heritage',
        'prompts': [
            "भोजपुरी लोक कथा सुनाइए। राजा-रानी की कहानी हो।",
            "भोजपुरी परीकथा बताइए। जादू-टोना वाली कहानी हो।",
            "गांव के लोगों की एक पुरानी कहानी सुनाइए।",
            "देवी-देवता की कहानी भोजपुरी में बताइए।",
            "भोजपुरी के धार्मिक त्योहारों की बात करिए।",
            "गांव के त्योहार और परंपराओं की कहानी सुनाइए।",
            "भोजपुरी संस्कृति में बुजुर्गों की भूमिका क्या है।",
            "भोजपुरी विवाह परंपराओं की बात करिए।",
            "गांव की पवित्र परंपराएं क्या होती हैं।",
            "भोजपुरी किंवदंतियां और लोक विश्वास बताइए।",
            "मेला और पर्वों का गांव में क्या महत्व है।",
            "भोजपुरी के संतों और महापुरुषों की कहानियां।",
            "गांव में प्राचीन परंपरा और रीति-रिवाज़।",
            "भोजपुरी कला और हस्तशिल्प की परंपरा।",
            "गांव की धरोहर और ऐतिहासिक स्थान।",
        ]
    },
    'b': {
        'name': 'Village Life & Daily Activities',
        'prompts': [
            "भोजपुरी गांव में किसान के दिन भर के काम की बात करिए।",
            "गांव की महिलाओं का रोज़मर्रा का जीवन कैसा होता है।",
            "भोजपुरी गांव में बाजार जाने की बात करिए।",
            "खेत में काम करने वाले लोगों का जीवन बताइए।",
            "गांव में परिवार का जीवन कैसा चलता है।",
            "भोजपुरी में मां और बेटे की बातचीत।",
            "गांव के बाज़ार में व्यापारियों की बातें।",
            "पति-पत्नी की रोज़मर्रा की बातचीत भोजपुरी में।",
            "गांव के स्कूल में बच्चों की कहानी।",
            "भोजपुरी में दादा-दादी और पोते की बातें।",
            "गांव के नाई, पंडित और दुकानदार की बातें।",
            "खेत के काम और फसल की बातें भोजपुरी में।",
            "गांव में जानवरों की देखभाल की बात करिए।",
            "महिलाओं का घर और खेत दोनों का काम।",
            "गांव के युवाओं का जीवन और सपने।",
        ]
    },
    'c': {
        'name': 'Food, Geography & Knowledge',
        'prompts': [
            "भोजपुरी व्यंजनों के बारे में बताइए। खिचुड़ी और दाल-भात कैसे बनाते हैं।",
            "भोजपुरी खाने की परंपराएं क्या हैं? त्योहारों पर क्या-क्या बनाते हैं।",
            "घर में रोटी-सब्जी बनाने की बात करिए।",
            "भोजपुरी क्षेत्र का भूगोल और जलवायु बताइए।",
            "भोजपुरी क्षेत्र के प्रसिद्ध स्थानों की बात करिए।",
            "गांव के आसपास की प्रकृति और वनस्पति।",
            "भोजपुरी कहावतें और मुहावरे सुनाइए।",
            "गांव के बुज़ुर्गों की बुद्धिमान बातें लिखिए।",
            "भोजपुरी में जीवन सीख देने वाली बातें।",
            "गांव के औषधीय पौधे और उनका उपयोग।",
            "भोजपुरी में पारंपरिक चिकित्सा की बातें।",
            "गांव के पशु-पक्षी और जीव-जंतु।",
            "भोजपुरी क्षेत्र की ऋतुओं की बात करिए।",
            "गांव की नदी और जल स्रोत।",
            "भोजपुरी में पढ़ाई और ज्ञान की महत्ता।",
        ]
    }
}

SYSTEM_PROMPT = """तुम भोजपुरी भाषा में लेखन करो। याद रखो:
- सिर्फ भोजपुरी का उपयोग करो, हिंदी या अंग्रेजी नहीं।
- भोजपुरी की असली बोली का उपयोग करो।
- 3-5 वाक्य या एक छोटा पैराग्राफ लिख।
- भोजपुरी के विशेष शब्द और व्याकरण का उपयोग करो।"""

# Bhojpuri-specific markers for quality gating
BHOJPURI_MARKERS = {
    'verbs': ['बा', 'बाड़ी', 'रहलें', 'लगल', 'गेल', 'होत', 'करल', 'सुनल', 'देखल'],
    'pronouns': ['हमरा', 'तोहरा', 'ओकरा', 'ऊ', 'नी', 'निस'],
    'suffixes': ['-ल', '-ीं', '-हिं', '-हु', '-स', '-खु'],
    'particles': ['बे', 'बेंन', 'बीं', 'बिया', 'अउ', 'औ', 'तौ'],
}


class SyntheticBhojpuriGenerator:
    """Generates Bhojpuri text via claude CLI, resumable cycle-based orchestrator."""

    def __init__(
        self,
        worker_id: str = 'a',
        data_dir: Optional[str] = None,
        generations_per_cycle: int = 50,
        cycle_delay_seconds: int = 30,
        target_tokens: int = 500_000_000,
    ):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")

        self.worker_id = worker_id
        self.data_dir = Path(data_dir)
        self.generations_per_cycle = generations_per_cycle
        self.cycle_delay_seconds = cycle_delay_seconds
        self.target_tokens = target_tokens
        self.stop_requested = False

        # Worker-specific state
        self.topic_config = WORKER_TOPICS.get(worker_id, WORKER_TOPICS['a'])
        self.log_file = self.data_dir / f"phase_synth_progress_{worker_id}.log"
        self.status_file = self.data_dir / f"phase_synth_status_{worker_id}.json"
        self.state_file = self.data_dir / f"synthetic_gen_state_{worker_id}.json"
        self.raw_dir = self.data_dir / f"synthetic_{worker_id}_raw"
        self.cleaned_dir = self.data_dir / f"synthetic_{worker_id}_cleaned"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()
        self.pre_tokens_at_start = self._get_corpus_tokens()

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _handle_sigterm(self, sig, frame):
        self._log_progress("INTERRUPTED by signal", silent=True)
        self.stop_requested = True

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            'next_batch_id': 0,
            'generations_attempted': 0,
            'generations_succeeded': 0,
            'texts_kept_after_cleaning': 0,
            'merged_lines': 0,
        }

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _log_progress(self, message: str, silent: bool = False):
        timestamp = datetime.now().isoformat(timespec='seconds')
        log_line = f"[{timestamp}] {message}"
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
        if not silent:
            logger.info(f"[Worker {self.worker_id}] {message}")

    def _get_corpus_tokens(self) -> int:
        total_size = 0
        for split in ('train', 'val', 'test'):
            split_file = self.data_dir / split / "bhoj.txt"
            if split_file.exists():
                total_size += split_file.stat().st_size
        return total_size // 4

    def _generate_bhojpuri(self, prompt: str) -> Optional[str]:
        """Call claude CLI to generate Bhojpuri text."""
        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    prompt,
                    "--system-prompt",
                    SYSTEM_PROMPT,
                    "--model",
                    "haiku",
                    "--output-format",
                    "text",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(f"Claude call failed: {result.stderr[:100]}")
                return None
            text = result.stdout.strip()
            return text if text else None
        except subprocess.TimeoutExpired:
            logger.warning("Claude call timed out")
            return None
        except Exception as e:
            logger.warning(f"Exception during generation: {e}")
            return None

    def _contains_bhojpuri_markers(self, text: str) -> bool:
        """Check if text contains sufficient Bhojpuri-specific markers."""
        if len(text) < 50:
            return False

        marker_count = 0
        text_lower = text.lower()

        for marker_list in BHOJPURI_MARKERS.values():
            for marker in marker_list:
                marker_count += text.count(marker.lower())

        # Need at least 2 markers per 100 chars to qualify as Bhojpuri
        marker_density = marker_count / (len(text) / 100)
        return marker_density >= 2.0

    def _save_raw_batch(self, batch_data: List[Dict]) -> int:
        """Save a batch of raw generated texts."""
        batch_id = self.state['next_batch_id']
        self.state['next_batch_id'] += 1

        output_file = self.raw_dir / f"synthetic_batch_{batch_id:06d}.jsonl"
        with open(output_file, 'w') as f:
            for item in batch_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Saved batch {batch_id} ({len(batch_data)} texts) to {output_file.name}")
        return batch_id

    def _archive_cleaned_dir(self):
        """Archive cleaned directory after merge."""
        if self.cleaned_dir.exists() and list(self.cleaned_dir.glob("*")):
            archive_dir = self.data_dir / f"synthetic_{self.worker_id}_cleaned_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(self.cleaned_dir), str(archive_dir))
            logger.info(f"Archived {self.cleaned_dir.name} to {archive_dir.name}")
            self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def run_cycle(self, cycle_num: int):
        """Execute one complete cycle of generation."""
        try:
            corpus_tokens_before = self._get_corpus_tokens()
            corpus_lines_before = sum(
                sum(1 for _ in open(self.data_dir / split / "bhoj.txt"))
                for split in ('train', 'val', 'test')
                if (self.data_dir / split / "bhoj.txt").exists()
            )

            self._log_progress(
                f"CYCLE {cycle_num} | Starting... (corpus: {corpus_tokens_before/1e6:.1f}M, {corpus_lines_before:,} lines)"
            )
            cycle_start_time = time.time()

            # Step 1: Generate Bhojpuri text
            logger.info("Step 1: Generating Bhojpuri text...")
            prompts = self.topic_config['prompts']
            batch_data = []

            for gen_idx in range(self.generations_per_cycle):
                prompt = prompts[gen_idx % len(prompts)]
                self.state['generations_attempted'] += 1

                text = self._generate_bhojpuri(prompt)
                if text and self._contains_bhojpuri_markers(text):
                    self.state['generations_succeeded'] += 1
                    batch_data.append({
                        'text': text,
                        'topic': self.topic_config['name'],
                        'generated_at': datetime.now().isoformat(),
                    })

                if (gen_idx + 1) % 10 == 0:
                    logger.info(f"  Generated {gen_idx + 1}/{self.generations_per_cycle}")

            if not batch_data:
                logger.warning("No text generated this cycle")
                self._log_progress(f"CYCLE {cycle_num:3d} | No output generated")
                return

            batch_id = self._save_raw_batch(batch_data)

            # Step 2: Clean generated text
            logger.info("Step 2: Cleaning generated text...")
            cleaner = BhojpuriDataCleaner(
                input_dir=str(self.raw_dir),
                output_dir=str(self.cleaned_dir)
            )
            cleaner.run_cleaning()
            report_file = self.cleaned_dir / "cleaning_report.json"
            texts_cleaned = 0
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                    texts_cleaned = report.get('total_kept', 0)
                    self.state['texts_kept_after_cleaning'] += texts_cleaned

            # Step 3: Deduplication
            logger.info("Step 3: Deduplicating against existing corpus...")
            with tempfile.TemporaryDirectory() as tmp:
                temp_dedup = Path(tmp) / f"synthetic_{self.worker_id}_dedup"
                temp_dedup.mkdir()
                deduplicate_with_existing_corpus(
                    data_dir=self.data_dir,
                    source_cleaned_dir=self.cleaned_dir,
                    output_cleaned_dir=temp_dedup
                )
                shutil.rmtree(self.cleaned_dir)
                shutil.move(str(temp_dedup), str(self.cleaned_dir))

            # Step 4: Merge into train/val/test
            logger.info("Step 4: Merging into splits...")
            merge_result = merge_ocr_into_splits(
                self.data_dir,
                accumulator_filename="bhoj.txt",
                split_filename="bhoj.txt",
                cleaned_dir_name=f"synthetic_{self.worker_id}_cleaned",
                temp_batch_filename=f"synthetic_{self.worker_id}_batch.txt"
            )
            merged_lines = merge_result.get('total_lines', 0)
            self.state['merged_lines'] += merged_lines
            self._archive_cleaned_dir()

            # Step 5: Update config
            logger.info("Step 5: Updating config...")
            update_config_bhojpuri(str(self.data_dir / "config.json"), self.data_dir)
            config_path = self.data_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                if "synthetic_generation" not in config:
                    config["synthetic_generation"] = {}
                config["synthetic_generation"][f"worker_{self.worker_id}"] = {
                    "topic": self.topic_config['name'],
                    "generations_attempted": self.state['generations_attempted'],
                    "generations_succeeded": self.state['generations_succeeded'],
                    "texts_after_cleaning": self.state['texts_kept_after_cleaning'],
                    "merged_lines": self.state['merged_lines'],
                    "last_cycle": cycle_num,
                    "last_updated": datetime.now().isoformat(),
                }
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

            # Logging
            corpus_tokens_after = self._get_corpus_tokens()
            corpus_lines_after = sum(
                sum(1 for _ in open(self.data_dir / split / "bhoj.txt"))
                for split in ('train', 'val', 'test')
                if (self.data_dir / split / "bhoj.txt").exists()
            )
            corpus_tokens_gained = corpus_tokens_after - corpus_tokens_before
            corpus_lines_gained = corpus_lines_after - corpus_lines_before
            progress_pct = 100.0 * corpus_tokens_after / self.target_tokens
            cycle_time = time.time() - cycle_start_time

            self._log_progress(
                f"CYCLE {cycle_num:3d} | generated:{self.state['generations_succeeded']:3d}/{self.state['generations_attempted']:3d} "
                f"cleaned:+{texts_cleaned:3d} merged:+{merged_lines:4d} | "
                f"CORPUS: {corpus_tokens_before/1e6:7.1f}M → {corpus_tokens_after/1e6:7.1f}M "
                f"(+{corpus_tokens_gained/1e6:6.2f}M, +{corpus_lines_gained:5d} lines) | "
                f"Progress: {progress_pct:5.1f}% ({corpus_tokens_after/1e6:7.1f}M/{self.target_tokens/1e6:7.1f}M) | "
                f"time:{cycle_time:6.1f}s"
            )

            status = {
                "updated_at": datetime.now().isoformat(timespec='seconds'),
                "worker_id": self.worker_id,
                "cycle_num": cycle_num,
                "corpus_tokens": corpus_tokens_after,
                "corpus_lines": corpus_lines_after,
                "target_tokens": self.target_tokens,
                "progress_percent": round(progress_pct, 2),
                "tokens_gained_this_cycle": corpus_tokens_gained,
                "lines_gained_this_cycle": corpus_lines_gained,
                "generations_attempted_total": self.state['generations_attempted'],
                "generations_succeeded_total": self.state['generations_succeeded'],
                "texts_kept_total": self.state['texts_kept_after_cleaning'],
                "merged_lines_total": self.state['merged_lines'],
                "cycle_time_seconds": round(cycle_time, 1)
            }

            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)

            self._save_state()

        except Exception as e:
            logger.error(f"Error in cycle {cycle_num}: {e}", exc_info=True)
            self._log_progress(f"ERROR in cycle {cycle_num}: {str(e)[:100]}")
            raise

    def run(self):
        """Main run loop."""
        self._log_progress(f"Synthetic Bhojpuri Generator (Worker {self.worker_id}) started")
        logger.info(f"Topic: {self.topic_config['name']}")

        cycle_num = 1
        while not self.stop_requested:
            try:
                self.run_cycle(cycle_num)
                cycle_num += 1

                if not self.stop_requested:
                    logger.info(f"Waiting {self.cycle_delay_seconds}s before next cycle...")
                    time.sleep(self.cycle_delay_seconds)

            except KeyboardInterrupt:
                self._log_progress("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Fatal error: {e}", exc_info=True)
                self._log_progress(f"FATAL ERROR: {str(e)[:200]}")
                raise

        self._log_progress("Synthetic generator stopped")
