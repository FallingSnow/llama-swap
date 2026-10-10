<script lang="ts">
  import type { Model } from "../lib/types";
  import { pendingLoads, onToggleLoad, unloadModel } from "../stores/modelLoad";
  import { Play, PowerOff, Loader2, Trash2 } from "@lucide/svelte";
  import {
    DropdownMenu,
    DropdownMenuTrigger,
    DropdownMenuContent,
    DropdownMenuItem,
  } from "$lib/components/ui/dropdown-menu";

  interface Props {
    model: Model;
    /** "md" for list rows (size-7), "sm" for the detail header (size-5). */
    size?: "md" | "sm";
  }

  let { model, size = "md" }: Props = $props();

  let btnSize = $derived(size === "sm" ? "size-5 rounded-sm" : "size-7 rounded-md");
  let iconSize = $derived(size === "sm" ? "size-3.5" : "size-4");
  let busy = $derived(model.state === "starting" || model.state === "stopping");
  let menuOpen = $state(false);

  function handleUnloadOption(force: boolean) {
    menuOpen = false;
    unloadModel(model, force);
  }
</script>

{#if model.state === "ready" && !busy}
  <DropdownMenu bind:open={menuOpen}>
    <DropdownMenuTrigger>
      <button
        type="button"
        class="text-muted-foreground hover:bg-accent hover:text-accent-foreground flex {btnSize} shrink-0 items-center justify-center disabled:opacity-50"
        title="Unload"
        aria-label="Unload model"
        disabled={busy}
      >
        <PowerOff class={iconSize} />
      </button>
    </DropdownMenuTrigger>
    <DropdownMenuContent align="end">
      <DropdownMenuItem onclick={() => handleUnloadOption(false)}>
        <PowerOff class="size-4" />
        Unload
      </DropdownMenuItem>
      <DropdownMenuItem variant="destructive" onclick={() => handleUnloadOption(true)}>
        <Trash2 class="size-4" />
        Kill
      </DropdownMenuItem>
    </DropdownMenuContent>
  </DropdownMenu>
{:else}
  <button
    type="button"
    class="text-muted-foreground hover:bg-accent hover:text-accent-foreground flex {btnSize} shrink-0 items-center justify-center disabled:opacity-50"
    title={$pendingLoads[model.id] ? "Cancel" : model.state === "suspended" ? "Wake" : "Load"}
    aria-label={$pendingLoads[model.id] ? "Cancel load" : model.state === "suspended" ? "Wake model" : "Load model"}
    disabled={busy}
    onclick={() => onToggleLoad(model)}
  >
    {#if $pendingLoads[model.id] && (model.state === "stopped" || model.state === "suspended")}
      <Loader2 class="{iconSize} animate-spin" />
    {:else if busy}
      <Loader2 class="{iconSize} animate-spin" />
    {:else}
      <Play class={iconSize} />
    {/if}
  </button>
{/if}
