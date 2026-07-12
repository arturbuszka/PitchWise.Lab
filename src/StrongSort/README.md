# StrongSort.Net

Pure C# implementation of **StrongSORT** (["StrongSORT: Make DeepSORT Great Again"](https://arxiv.org/abs/2202.13514)) —
a multi-object tracker combining an NSA Kalman filter, EMA-smoothed appearance embeddings
and Hungarian assignment. No native or ML dependencies: bring your own detector and
(optionally) ReID embeddings.

```csharp
using StrongSort;

var tracker = new StrongSortTracker(new StrongSortOptions
{
    MaxAppearanceDistance = 0.5f,
    MaxAge = 300,
});

foreach (var frame in video)
{
    var observations = detector.Detect(frame)
        .Select(d => new Observation(d.Left, d.Top, d.Width, d.Height, d.Confidence,
                                     Feature: reid.Embed(frame, d), Tag: d.Id));
    IReadOnlyList<TrackResult> tracks = tracker.Update(observations.ToList());
    // tracks: stable TrackId, Kalman-smoothed BboxLtwh, Tag of the matched observation
}
```

- Works without embeddings too (`Feature: null`) — matching then falls back to the IoU stage.
- `Tag` is an arbitrary payload (e.g. your detection id) returned on the matched track.
- Defaults follow the SoccerNet game-state baseline; original paper defaults are documented
  on `StrongSortOptions`.
