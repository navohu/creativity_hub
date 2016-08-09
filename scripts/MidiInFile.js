class MidiInFile{
	constructor(outStream, inFile){
		this.raw_in = RawInstreamFile(infile);
		this.parser = MidiFileParser(this.raw_in, outStream);
	}

	read() {
		p = this.parser
		p.parseMThdChunk()
		p.parseMTrkChunks()
	}
}