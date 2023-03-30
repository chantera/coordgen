import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

type Span = [number, number];

interface Coord {
  text: string;
  cc: Span;
  conjuncts: Span[];
}

type SelectorProps = {
  onSubmit: (text: string, span: Span) => void;
  onReset: () => void;
  disabled: boolean;
}

const Selector: React.FC<SelectorProps> = (props) => {
  const [input, setInput] = useState<string>("");
  const [span, setSpan] = useState<Span | null>(null);

  const handleSelect = (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
    const start = (e.target as HTMLTextAreaElement).selectionStart;
    const end = (e.target as HTMLTextAreaElement).selectionEnd;
    if (end > start) {
      setSpan([start, end]);
    }
  }

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!span || props.disabled) {
      return;
    }
    props.onSubmit(input, span);
  }

  const handleReset = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    setText("");
    props.onReset();
  }

  const setText = (v: string) => {
    setInput(v);
    setSpan(null);
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-y-4 gap-x-8">
        <div className="sm:col-span-4">
          <label htmlFor="input" className="block label">Input</label>
          <div className="mt-2">
            <textarea id="input" value={input} onChange={(e) => setText(e.target.value)} onSelect={handleSelect} className="block w-full input" rows={4} />
          </div>
        </div>
        <div className="sm:col-span-2">
          <code className="block h-10 preview truncate">{span && `(${span[0]}, ${span[1]}): ${input.substring(span[0], span[1])}`}</code>
        </div>
        <div>
          <button type="submit" disabled={!span || props.disabled} className="block w-full btn btn-primary">Generate</button>
        </div>
        <div>
          <button type="reset" onClick={handleReset} className="block w-full btn">Reset</button>
        </div>
      </div>
    </form>
  );
}

const List = (props: { coords: Coord[] }) => {
  const results = props.coords.map((coord, i) => {
    const [pre, post] = coord.conjuncts;
    const head = coord.text.substring(0, pre[0]);
    const conj1 = coord.text.substring(pre[0], pre[1]);
    const cc = coord.text.substring(pre[1], post[0]);
    const conj2 = coord.text.substring(post[0], post[1]);
    const tail = coord.text.substring(post[1]);

    return (
      <li key={i} className="rounded-md bg-gray-100 px-3.5 py-2.5 mt-4 text-gray-900">
        {head}<u>[{conj1}]</u><b>{cc}</b>[{conj2}]{tail}
      </li>
    );
  });

  return (
    <ol>{results}</ol>
  );
}

const App = () => {
  const [coords, setCoords] = useState<Coord[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = (text: string, span: Span) => {
    setLoading(true);
    axios
      .post("http://localhost:8000/generate", {
        text: text, start: span[0], end: span[1]
      })
      .then((res) => {
        const coord: Coord = {
          text: res.data.text,
          cc: [res.data.cc.start, res.data.cc.end],
          conjuncts: res.data.conjuncts.map((x: any) => [x.start, x.end])
        };
        setCoords(coords.concat([coord]));
      })
      .catch((err) => {
        console.log(err);
        alert("Oops! Something went wrong.");
      })
      .finally(() => {
        setLoading(false);
      });
  }

  return (
    <div className="min-h-full">
      <header className="bg-white shadow-sm">
        <div className="mx-auto max-w-4xl py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Coordgen</h1>
        </div>
      </header>
      <main>
        <div className="mx-auto max-w-4xl py-6 sm:px-6 lg:px-8">
          <div className="bg-white rounded py-6 px-4 shadow mb-8">
            <Selector onSubmit={handleSubmit} onReset={() => setCoords([])} disabled={loading} />
          </div>
          <div className="bg-white rounded py-6 px-4 shadow" style={{ display: coords.length ? "block" : "none" }}>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Results</h2>
            <List coords={coords} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
