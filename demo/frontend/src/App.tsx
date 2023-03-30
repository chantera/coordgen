import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

interface Coord {
  text: string;
  cc: [number, number];
  conjuncts: [number, number][];
}

const Selector = (props: any) => {
  const [input, setInput] = useState<string>("");
  const [span, setSpan] = useState<[number, number] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSelect = (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
    if (!(e.target instanceof HTMLTextAreaElement)) {
      return;
    }
    const start = e.target.selectionStart;
    const end = e.target.selectionEnd;
    if (end > start) {
      setSpan([start, end]);
    }
  }

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    // console.log(input);
    // console.log(span);
    e.preventDefault();
    if (!span || loading) {
      return;
    }
    setLoading(true);
    axios
      .post("http://localhost:8000/generate", {
        text: input, start: span[0], end: span[1]
      })
      .then((res) => {
        const coord: Coord = {
          text: res.data.text,
          cc: [res.data.cc[0], res.data.cc[1]],
          conjuncts: res.data.conjuncts.map((x: any) => [x.start, x.end])
        };
        props.append(coord);
      })
      .catch((err) => {
        console.log(err);
        alert("Oops! Something went wrong.");
      })
      .finally(() => {
        setLoading(false);
      });
  }

  const setText = (v: string) => {
    setInput(v);
    setSpan(null);
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded py-6 px-4 mb-1 shadow">
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
          <button type="submit" disabled={!span || loading} className="block w-full btn btn-primary">Generate</button>
        </div>
        <div>
          <button type="reset" onClick={() => { setText(""); props.clear(); }} className="block w-full btn">Reset</button>
        </div>
      </div>
    </form>
  );
}

const App = () => {
  const [coords, setCoords] = useState<Coord[]>([]);

  const results = coords.map((coord, i) => {
    return (
      <li key={i}>
        <p>{coord.text}</p>
      </li>
    );
  });

  return (
    <div className="min-h-full">
      <header className="bg-white shadow-sm">
        <div className="mx-auto max-w-4xl py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">Coordgen</h1>
        </div>
      </header>
      <main>
        <div className="mx-auto max-w-4xl py-6 sm:px-6 lg:px-8">
          <Selector
            append={(coord: Coord) => setCoords(coords.concat([coord]))}
            clear={() => setCoords([])}
          />
          <ol>{results}</ol>
        </div>
      </main>
    </div>
  );
}

export default App;
