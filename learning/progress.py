"""Small versioned learning progress record, serializable to a user-owned JSON file."""
import json
from datetime import datetime, timezone

VERSION = 1

def empty_progress():
    return {'schema_version': VERSION, 'completed': [], 'quiz_scores': {}, 'exercise_notes': {}}

def mark_complete(progress: dict, lesson: str) -> dict:
    result = dict(progress)
    result['completed'] = sorted(set(progress.get('completed', [])) | {lesson})
    return result

def record_quiz(progress: dict, lesson: str, correct: int, total: int) -> dict:
    if not 0 <= correct <= total or total < 1:
        raise ValueError('Invalid quiz score')
    result = dict(progress)
    result['quiz_scores'] = {**progress.get('quiz_scores', {}), lesson: {'correct': correct, 'total': total,
                         'recorded_utc': datetime.now(timezone.utc).isoformat()}}
    return result

def load_progress(raw: bytes | str) -> dict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or parsed.get('schema_version') != VERSION:
        raise ValueError('Unsupported progress format/version')
    completed = parsed.get('completed', [])
    scores = parsed.get('quiz_scores', {})
    notes = parsed.get('exercise_notes', {})
    if not isinstance(completed, list) or not all(isinstance(x, str) for x in completed) or not isinstance(scores, dict) or not isinstance(notes, dict):
        raise ValueError('Malformed progress data')
    return {'schema_version': VERSION, 'completed': completed, 'quiz_scores': scores, 'exercise_notes': notes}

def export_progress(progress: dict) -> str:
    return json.dumps(progress, indent=2, sort_keys=True)
