"""
MIDI Rendering Module
Converts musical arrangements to MIDI files for playback and export.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import struct
import time

try:
    import mido
    from mido import MidiFile, MidiTrack, Message
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    import music21
    from music21 import stream, note, chord, meter, tempo, key, instrument
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False


class MidiRenderer:
    """Renders musical arrangements to MIDI files."""
    
    def __init__(self):
        """Initialize MIDI renderer."""
        self.logger = logging.getLogger(__name__)
        
        # MIDI constants
        self.ticks_per_beat = 480
        self.default_velocity = 80
        
    def render_to_midi(
        self,
        arrangement: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Render musical arrangement to MIDI file.
        
        Args:
            arrangement: Musical arrangement data
            output_path: Path for output MIDI file
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            if HAS_MIDO:
                return self._render_with_mido(arrangement, output_path)
            elif HAS_MUSIC21:
                return self._render_with_music21(arrangement, output_path)
            else:
                return self._render_basic_midi(arrangement, output_path)
        except Exception as e:
            self.logger.error(f"Error rendering MIDI: {e}")
            return False
    
    def _render_with_mido(self, arrangement: Dict[str, Any], output_path: str) -> bool:
        """Render using mido library."""
        
        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        
        # Set tempo
        tempo_bpm = arrangement.get('tempo', 120)
        microseconds_per_beat = int(60000000 / tempo_bpm)
        
        tracks = arrangement.get('tracks', {})
        
        for track_name, track_data in tracks.items():
            
            # Create MIDI track
            track = MidiTrack()
            mid.tracks.append(track)
            
            # Set instrument program
            instrument_info = track_data.get('instrument', {})
            program = instrument_info.get('program', 0)
            channel = instrument_info.get('channel', 1) - 1  # MIDI channels are 0-based
            
            # Add program change
            if program != 128:  # Not drums
                track.append(Message('program_change', channel=channel, program=program, time=0))
            
            # Add tempo (only on first track)
            if len(mid.tracks) == 1:
                track.append(Message('set_tempo', tempo=microseconds_per_beat, time=0))
            
            # Convert notes to MIDI events
            if track_data.get('type') == 'melody':
                self._add_melody_to_track(track, track_data['notes'], channel, tempo_bpm)
            elif track_data.get('type') == 'chords':
                self._add_chords_to_track(track, track_data['notes'], channel, tempo_bpm)
            elif track_data.get('type') == 'bass':
                self._add_bass_to_track(track, track_data['notes'], channel, tempo_bpm)
            elif track_data.get('type') == 'drums':
                self._add_drums_to_track(track, track_data['notes'], 9)  # Channel 10 (0-based 9)
        
        # Save MIDI file
        mid.save(output_path)
        self.logger.info(f"MIDI file saved: {output_path}")
        return True
    
    def _add_melody_to_track(
        self, 
        track: 'MidiTrack', 
        notes: List[int], 
        channel: int, 
        tempo: int
    ):
        """Add melody notes to MIDI track."""
        
        beat_duration = self.ticks_per_beat  # Quarter note duration
        current_time = 0
        
        for i, note_value in enumerate(notes):
            if 0 <= note_value <= 127:  # Valid MIDI note range
                
                # Note on
                track.append(Message(
                    'note_on',
                    channel=channel,
                    note=60 + note_value,  # Base on middle C
                    velocity=self.default_velocity,
                    time=current_time
                ))
                
                # Note duration (quarter note)
                note_duration = beat_duration
                
                # Note off
                track.append(Message(
                    'note_off',
                    channel=channel,
                    note=60 + note_value,
                    velocity=0,
                    time=note_duration
                ))
                
                current_time = 0  # Next note starts immediately after this one
    
    def _add_chords_to_track(
        self, 
        track: 'MidiTrack', 
        chord_notes: List[List[int]], 
        channel: int, 
        tempo: int
    ):
        """Add chord progression to MIDI track."""
        
        chord_duration = self.ticks_per_beat * 4  # Whole note duration
        current_time = 0
        
        for chord in chord_notes:
            # Chord on - all notes start together
            for j, midi_note in enumerate(chord):
                if 0 <= midi_note <= 127:
                    track.append(Message(
                        'note_on',
                        channel=channel,
                        note=midi_note,
                        velocity=self.default_velocity - 10,  # Slightly softer than melody
                        time=current_time if j == 0 else 0  # Only first note has timing
                    ))
            
            # Chord off - all notes end together
            for j, midi_note in enumerate(chord):
                if 0 <= midi_note <= 127:
                    track.append(Message(
                        'note_off',
                        channel=channel,
                        note=midi_note,
                        velocity=0,
                        time=chord_duration if j == 0 else 0  # Only first note has timing
                    ))
            
            current_time = 0
    
    def _add_bass_to_track(
        self, 
        track: 'MidiTrack', 
        bass_notes: List[int], 
        channel: int, 
        tempo: int
    ):
        """Add bass line to MIDI track."""
        
        note_duration = self.ticks_per_beat * 2  # Half note duration
        current_time = 0
        
        for bass_note in bass_notes:
            if 0 <= bass_note <= 127:
                
                # Bass note on
                track.append(Message(
                    'note_on',
                    channel=channel,
                    note=bass_note,
                    velocity=self.default_velocity + 10,  # Slightly louder
                    time=current_time
                ))
                
                # Bass note off
                track.append(Message(
                    'note_off',
                    channel=channel,
                    note=bass_note,
                    velocity=0,
                    time=note_duration
                ))
                
                current_time = 0
    
    def _add_drums_to_track(
        self, 
        track: 'MidiTrack', 
        drum_events: List[Dict[str, Any]], 
        channel: int
    ):
        """Add drum pattern to MIDI track."""
        
        # Sort events by time
        sorted_events = sorted(drum_events, key=lambda x: x.get('time', 0))
        
        current_tick = 0
        
        for event in sorted_events:
            event_time = int(event.get('time', 0) * self.ticks_per_beat)
            note = event.get('note', 36)
            velocity = event.get('velocity', 80)
            duration = int(event.get('duration', 0.25) * self.ticks_per_beat)
            
            # Calculate time delta
            time_delta = event_time - current_tick
            
            # Drum hit
            track.append(Message(
                'note_on',
                channel=channel,
                note=note,
                velocity=velocity,
                time=time_delta
            ))
            
            # Drum release
            track.append(Message(
                'note_off',
                channel=channel,
                note=note,
                velocity=0,
                time=duration
            ))
            
            current_tick = event_time
    
    def _render_with_music21(self, arrangement: Dict[str, Any], output_path: str) -> bool:
        """Render using music21 library."""
        
        # Create a score
        score = stream.Score()
        
        # Add metadata
        score.insert(0, meter.TimeSignature('4/4'))
        score.insert(0, tempo.TempoIndication(number=arrangement.get('tempo', 120)))
        
        key_sig = arrangement.get('key', 0)
        mode = arrangement.get('mode', 'major')
        score.insert(0, key.KeySignature(key_sig))
        
        tracks = arrangement.get('tracks', {})
        
        for track_name, track_data in tracks.items():
            
            # Create part
            part = stream.Part()
            
            # Set instrument
            instrument_info = track_data.get('instrument', {})
            if 'name' in instrument_info:
                # Add instrument (simplified)
                part.insert(0, instrument.Piano())  # Default to piano
            
            # Add notes
            if track_data.get('type') == 'melody':
                self._add_melody_music21(part, track_data['notes'])
            elif track_data.get('type') == 'chords':
                self._add_chords_music21(part, track_data['notes'])
            
            score.append(part)
        
        # Write MIDI file
        score.write('midi', fp=output_path)
        self.logger.info(f"MIDI file saved with music21: {output_path}")
        return True
    
    def _add_melody_music21(self, part: 'stream.Part', notes: List[int]):
        """Add melody using music21."""
        
        for note_value in notes:
            if 0 <= note_value <= 127:
                n = note.Note(midi=60 + note_value)
                n.duration.quarterLength = 1.0  # Quarter note
                part.append(n)
    
    def _add_chords_music21(self, part: 'stream.Part', chord_notes: List[List[int]]):
        """Add chords using music21."""
        
        for chord_midi_notes in chord_notes:
            if chord_midi_notes:
                c = chord.Chord([note.Note(midi=mn) for mn in chord_midi_notes])
                c.duration.quarterLength = 4.0  # Whole note
                part.append(c)
    
    def _render_basic_midi(self, arrangement: Dict[str, Any], output_path: str) -> bool:
        """Basic MIDI rendering without external libraries."""
        
        self.logger.warning("Using basic MIDI rendering - limited functionality")
        
        # Create a simple MIDI file structure
        midi_data = self._create_basic_midi_structure(arrangement)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(midi_data)
        
        self.logger.info(f"Basic MIDI file saved: {output_path}")
        return True
    
    def _create_basic_midi_structure(self, arrangement: Dict[str, Any]) -> bytes:
        """Create basic MIDI file structure manually."""
        
        # MIDI Header
        header = b'MThd'  # Header chunk type
        header += struct.pack('>I', 6)  # Header length
        header += struct.pack('>H', 0)  # Format type 0
        header += struct.pack('>H', 1)  # Number of tracks
        header += struct.pack('>H', self.ticks_per_beat)  # Ticks per quarter note
        
        # Track Header
        track_data = b''
        
        # Set tempo
        tempo_bpm = arrangement.get('tempo', 120)
        microseconds_per_beat = int(60000000 / tempo_bpm)
        
        # Delta time (0) + Meta event (Set Tempo)
        track_data += b'\x00\xFF\x51\x03'
        track_data += struct.pack('>I', microseconds_per_beat)[1:]  # Remove first byte
        
        # Add some basic melody notes (simplified)
        tracks = arrangement.get('tracks', {})
        if 'melody' in tracks:
            melody_notes = tracks['melody'].get('notes', [])
            
            for note_value in melody_notes[:16]:  # Limit to first 16 notes
                if 0 <= note_value <= 127:
                    midi_note = 60 + note_value
                    
                    # Note on
                    track_data += b'\x00'  # Delta time
                    track_data += struct.pack('B', 0x90)  # Note on, channel 0
                    track_data += struct.pack('B', midi_note)
                    track_data += struct.pack('B', 64)  # Velocity
                    
                    # Note off (after quarter note)
                    track_data += struct.pack('B', self.ticks_per_beat)  # Delta time
                    track_data += struct.pack('B', 0x80)  # Note off, channel 0
                    track_data += struct.pack('B', midi_note)
                    track_data += struct.pack('B', 0)  # Velocity
        
        # End of track
        track_data += b'\x00\xFF\x2F\x00'
        
        # Track chunk
        track_chunk = b'MTrk'
        track_chunk += struct.pack('>I', len(track_data))
        track_chunk += track_data
        
        return header + track_chunk


class MidiAnalyzer:
    """Analyzes and validates MIDI files."""
    
    def __init__(self):
        """Initialize MIDI analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_midi_file(self, midi_path: str) -> Dict[str, Any]:
        """
        Analyze a MIDI file and return information about it.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dictionary with MIDI file analysis
        """
        
        analysis = {
            'valid': False,
            'tracks': 0,
            'duration': 0.0,
            'notes': 0,
            'tempo': 120,
            'time_signature': [4, 4],
            'key_signature': 'C major'
        }
        
        try:
            if HAS_MIDO:
                analysis = self._analyze_with_mido(midi_path)
            elif HAS_MUSIC21:
                analysis = self._analyze_with_music21(midi_path)
            else:
                analysis = self._analyze_basic(midi_path)
                
            analysis['valid'] = True
            
        except Exception as e:
            self.logger.error(f"Error analyzing MIDI file: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_with_mido(self, midi_path: str) -> Dict[str, Any]:
        """Analyze MIDI file using mido."""
        
        mid = MidiFile(midi_path)
        
        analysis = {
            'tracks': len(mid.tracks),
            'ticks_per_beat': mid.ticks_per_beat,
            'notes': 0,
            'duration': 0.0,
            'tempo': 120
        }
        
        total_ticks = 0
        current_tempo = 500000  # Default microseconds per beat
        
        for track in mid.tracks:
            track_ticks = 0
            
            for msg in track:
                track_ticks += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    analysis['notes'] += 1
                elif msg.type == 'set_tempo':
                    current_tempo = msg.tempo
                    analysis['tempo'] = int(60000000 / current_tempo)
            
            total_ticks = max(total_ticks, track_ticks)
        
        # Calculate duration
        if mid.ticks_per_beat > 0:
            beats = total_ticks / mid.ticks_per_beat
            analysis['duration'] = beats * (current_tempo / 1000000)  # Convert to seconds
        
        return analysis
    
    def _analyze_with_music21(self, midi_path: str) -> Dict[str, Any]:
        """Analyze MIDI file using music21."""
        
        score = music21.converter.parse(midi_path)
        
        analysis = {
            'tracks': len(score.parts),
            'notes': 0,
            'duration': float(score.duration.quarterLength),
            'tempo': 120
        }
        
        # Count notes
        for part in score.parts:
            for element in part.flat.notes:
                if element.isNote:
                    analysis['notes'] += 1
                elif element.isChord:
                    analysis['notes'] += len(element.pitches)
        
        # Get tempo
        tempo_marks = score.flat.getElementsByClass(music21.tempo.TempoIndication)
        if tempo_marks:
            analysis['tempo'] = int(tempo_marks[0].number)
        
        return analysis
    
    def _analyze_basic(self, midi_path: str) -> Dict[str, Any]:
        """Basic MIDI file analysis."""
        
        analysis = {
            'tracks': 1,
            'notes': 0,
            'duration': 0.0,
            'tempo': 120
        }
        
        # Just check if file exists and get size
        midi_path = Path(midi_path)
        if midi_path.exists():
            analysis['file_size'] = midi_path.stat().st_size
            analysis['notes'] = analysis['file_size'] // 10  # Rough estimate
        
        return analysis
    
    def validate_midi_output(self, midi_path: str) -> bool:
        """
        Validate that a MIDI file was created correctly.
        
        Args:
            midi_path: Path to MIDI file to validate
            
        Returns:
            True if valid, False otherwise
        """
        
        try:
            analysis = self.analyze_midi_file(midi_path)
            
            # Basic validation checks
            if not analysis['valid']:
                return False
            
            if analysis['tracks'] == 0:
                self.logger.warning("MIDI file has no tracks")
                return False
            
            if analysis['notes'] == 0:
                self.logger.warning("MIDI file has no notes")
                return False
            
            if analysis['duration'] <= 0:
                self.logger.warning("MIDI file has zero duration")
                return False
            
            self.logger.info(f"MIDI file validation successful: {analysis}")
            return True
            
        except Exception as e:
            self.logger.error(f"MIDI validation failed: {e}")
            return False


# Convenience function for easy MIDI rendering
def render_arrangement_to_midi(
    arrangement: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Convenience function to render arrangement to MIDI.
    
    Args:
        arrangement: Musical arrangement data
        output_path: Output MIDI file path
        
    Returns:
        True if successful, False otherwise
    """
    
    renderer = MidiRenderer()
    return renderer.render_to_midi(arrangement, output_path)